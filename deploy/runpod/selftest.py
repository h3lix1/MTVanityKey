#!/usr/bin/env python3
"""Self-test and test client of the handler's result encryption.

    python3 selftest.py
        ECIES round trip, tamper and wrong-key rejection, client_pubkey and
        job-input validation (no GPU needed).
    python3 selftest.py input meshtastic '!c0de' [HITS] > test_input.json
    python3 selftest.py input meshcore BEEF > test_input.json
        A job input with a fresh client key; the private key goes to
        client-key.pem (mode 600) for `decrypt`.
    python3 selftest.py decrypt client-key.pem < output
        Decrypt a job output (JSON, or the local test mode's
        "Job result: {...}" log line), print the hits and check every one
        independently: X25519 (OpenSSL) and zlib's CRC-32 for Meshtastic,
        a pure-Python Ed25519 base-point multiplication for MeshCore.
"""

import ast
import json
import os
import sys
import zlib

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, x25519

import ecies

KEY_FILE = "client-key.pem"


def new_client():
    private = ec.generate_private_key(ec.SECP256R1())
    return private, ecies.b64url_encode(ecies.raw_point(private.public_key()))


def selftest():
    private, public_b64 = new_client()
    client = ecies.parse_client_pubkey(public_b64)
    message = json.dumps({"hits": [{"node_id": "!c0de0001"}]}).encode()
    enc = ecies.encrypt(client, message)
    assert enc["alg"] == ecies.ALG
    assert all("=" not in enc[k] for k in ("epk", "iv", "ct")), "unpadded base64url"
    assert len(ecies.b64url_decode(enc["epk"])) == 65 and len(ecies.b64url_decode(enc["iv"])) == 12
    assert len(ecies.b64url_decode(enc["ct"])) == len(message) + 16
    assert ecies.decrypt(private, enc) == message
    # Fresh ephemeral key and IV every time.
    again = ecies.encrypt(client, message)
    assert again["epk"] != enc["epk"] and again["iv"] != enc["iv"]

    # Any modified field, or another client's key, fails authentication.
    for field in ("ct", "iv"):
        raw = bytearray(ecies.b64url_decode(enc[field]))
        raw[0] ^= 1
        try:
            ecies.decrypt(private, {**enc, field: ecies.b64url_encode(bytes(raw))})
            raise AssertionError(f"tampered {field} accepted")
        except InvalidTag:
            pass
    other, _ = new_client()
    try:
        ecies.decrypt(other, enc)
        raise AssertionError("wrong key accepted")
    except InvalidTag:
        pass
    _, other_pub = new_client()
    try:
        ecies.decrypt(private, {**enc, "epk": other_pub})
        raise AssertionError("substituted epk accepted")
    except InvalidTag:
        pass

    # client_pubkey validation (padded base64url is accepted too).
    assert ecies.parse_client_pubkey(public_b64 + "=")
    for bad in (None, "", "!!", ecies.b64url_encode(b"\x04" + b"\x00" * 64),
                ecies.b64url_encode(b"\x02" + b"\x01" * 32), public_b64[:-4]):
        try:
            ecies.parse_client_pubkey(bad)
            raise AssertionError(f"bad client_pubkey accepted: {bad!r}")
        except ValueError:
            pass

    # Job-input validation of the handler (no server is started on import).
    import handler

    path, _ = handler.local_path({"flow": "meshtastic", "pattern": "!dead,beef", "hits": 2, "client_pubkey": public_b64})
    assert path == "/v1/meshtastic?pattern=%21dead%2Cbeef&hits=2", path
    path, _ = handler.local_path({"flow": "meshcore", "prefix": "BEEF", "hits": 3, "client_pubkey": public_b64})
    assert path == "/v1/meshcore?prefix=BEEF", path
    for bad, needle in (
        ({"flow": "meshtastic", "pattern": "dead"}, "client_pubkey is required"),
        ({"flow": "meshtastic", "pattern": "dead", "client_pubkey": "AAAA"}, "65-byte"),
        ({"flow": "reticulum", "pattern": "1", "client_pubkey": public_b64}, "flow must be"),
        ({"flow": "meshtastic", "pattern": 7, "client_pubkey": public_b64}, "pattern must be"),
        ({"flow": "meshtastic", "pattern": "dead", "hits": True, "client_pubkey": public_b64}, "hits"),
        ({"flow": "meshtastic", "pattern": "dead", "hits": 1.5, "client_pubkey": public_b64}, "hits"),
        ({"flow": "meshcore", "client_pubkey": public_b64}, "prefix must be"),
        ("not an object", "JSON object"),
    ):
        try:
            handler.local_path(bad)
            raise AssertionError(f"bad input accepted: {bad!r}")
        except ValueError as error:
            assert needle in str(error), (needle, str(error))
    print("ECIES and input validation self-test ok")


def make_input(flow, pattern, hits=None):
    private, public_b64 = new_client()
    fd = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                      serialization.NoEncryption()))
    job = {"flow": flow, "client_pubkey": public_b64}
    if flow == "meshtastic":
        job["pattern"] = pattern
        job["hits"] = int(hits or 1)
    else:
        job["prefix"] = pattern
    print(json.dumps({"input": job}))


# Ed25519 public key of a clamped scalar (RFC 8032 arithmetic, affine).
P = 2**255 - 19
D = -121665 * pow(121666, P - 2, P) % P
BASE = (15112221349535400772501151409588531511454012693041857206046113283949847762202,
        4 * pow(5, P - 2, P) % P)


def _add(a, b):
    (x1, y1), (x2, y2) = a, b
    t = D * x1 * x2 * y1 * y2 % P
    return ((x1 * y2 + x2 * y1) * pow(1 + t, P - 2, P) % P, (y1 * y2 + x1 * x2) * pow(1 - t, P - 2, P) % P)


def ed25519_public(scalar: bytes) -> bytes:
    k, r, q = int.from_bytes(scalar, "little"), (0, 1), BASE
    while k:
        if k & 1:
            r = _add(r, q)
        q, k = _add(q, q), k >> 1
    return (r[1] | (r[0] & 1) << 255).to_bytes(32, "little")


def read_output(text):
    text = text.strip()
    if "Job result: " in text:
        text = text[text.rindex("Job result: ") + len("Job result: "):].splitlines()[0]
        result = ast.literal_eval(text)
    else:
        result = json.loads(text)
    return result.get("output", result)


def decrypt(key_file):
    with open(key_file, "rb") as f:
        private = serialization.load_pem_private_key(f.read(), password=None)
    output = read_output(sys.stdin.read())
    hits = json.loads(ecies.decrypt(private, output["enc"]))["hits"]
    print(json.dumps({k: v for k, v in output.items() if k != "enc"}))
    for hit in hits:
        if "node_id" in hit:
            secret = bytes.fromhex(hit["private_key_hex"])
            assert secret[0] & 7 == 0 and secret[31] & 0xC0 == 0x40, "clamped"
            public = x25519.X25519PrivateKey.from_private_bytes(secret).public_key().public_bytes_raw()
            assert public.hex() == hit["public_key_hex"], "public key"
            assert f"!{zlib.crc32(public):08x}" == hit["node_id"], "node id"
            print(f"verified meshtastic {hit['node_id']} ({hit['source']})")
        else:
            private_key = bytes.fromhex(hit["private_key"])
            assert len(private_key) == 64
            assert ed25519_public(private_key[:32]).hex().upper() == hit["public_key"], "public key"
            print(f"verified meshcore {hit['public_key'][:16]}... ({hit['source']})")
    print(f"{len(hits)} hit(s) decrypted and verified")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selftest()
    elif sys.argv[1] == "input" and len(sys.argv) in (4, 5):
        make_input(*sys.argv[2:])
    elif sys.argv[1] == "decrypt" and len(sys.argv) == 3:
        decrypt(sys.argv[2])
    else:
        sys.exit(__doc__)
