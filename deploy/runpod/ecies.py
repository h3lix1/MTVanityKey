"""Result encryption of the RunPod handler (ECIES to a browser's P-256 key).

The client sends a fresh P-256 public key (65-byte uncompressed point,
base64url) with every job; the result is encrypted to it so the plaintext
private keys never reach the job store:

    shared = ECDH(ephemeral, client)                      (32 bytes)
    key    = HKDF-SHA256(ikm=shared, salt=epk || client_pub,
                         info="meshkeys.io/v1 result", length=32)
    ct     = AES-256-GCM(key, iv=12 random bytes, aad="meshkeys.io/v1")

`ct` is the ciphertext with the 16-byte tag appended (WebCrypto's layout).
All byte strings are unpadded base64url. `decrypt` is the reference client
used by the self-test.
"""

import base64
import os

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

ALG = "ECDH-P256+HKDF-SHA256+AES-256-GCM"
INFO = b"meshkeys.io/v1 result"
AAD = b"meshkeys.io/v1"


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64url_decode(text: str) -> bytes:
    if not isinstance(text, str):
        raise ValueError("expected a base64url string")
    text = text.rstrip("=")
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def raw_point(key: ec.EllipticCurvePublicKey) -> bytes:
    return key.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)


def parse_client_pubkey(text) -> ec.EllipticCurvePublicKey:
    """The client's P-256 public key from its base64url raw point (65 bytes,
    0x04 || x || y); raises ValueError unless it is a valid curve point."""
    try:
        raw = b64url_decode(text)
    except Exception:
        raise ValueError("client_pubkey must be base64url") from None
    if len(raw) != 65 or raw[0] != 4:
        raise ValueError("client_pubkey must be a 65-byte uncompressed P-256 point")
    try:
        return ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), raw)
    except ValueError:
        raise ValueError("client_pubkey is not a point on P-256") from None


def _key(shared: bytes, epk_raw: bytes, client_raw: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=epk_raw + client_raw, info=INFO).derive(shared)


def encrypt(client: ec.EllipticCurvePublicKey, plaintext: bytes) -> dict:
    """The `enc` object of a job's output."""
    ephemeral = ec.generate_private_key(ec.SECP256R1())
    epk_raw = raw_point(ephemeral.public_key())
    key = _key(ephemeral.exchange(ec.ECDH(), client), epk_raw, raw_point(client))
    iv = os.urandom(12)
    ct = AESGCM(key).encrypt(iv, plaintext, AAD)
    return {"alg": ALG, "epk": b64url_encode(epk_raw), "iv": b64url_encode(iv), "ct": b64url_encode(ct)}


def decrypt(client_private: ec.EllipticCurvePrivateKey, enc: dict) -> bytes:
    """Reference client: open an `enc` object with the job's private key."""
    if enc.get("alg") != ALG:
        raise ValueError("unsupported alg")
    epk_raw = b64url_decode(enc["epk"])
    epk = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), epk_raw)
    key = _key(client_private.exchange(ec.ECDH(), epk), epk_raw, raw_point(client_private.public_key()))
    return AESGCM(key).decrypt(b64url_decode(enc["iv"]), b64url_decode(enc["ct"]), AAD)
