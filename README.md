# MTVanityKey

Vanity key generator for mesh radios, on CPU, Apple Metal and NVIDIA CUDA,
with a command-line tool and an HTTP JSON API with a job queue.

Two independent flows share one search engine:

* **Meshtastic**: choose your node ID. Since firmware 2.8 a node's number is
  derived from its X25519 identity key, so a key whose node ID starts with
  `!dc80`, spells a word or matches a list of candidates can be found by
  grinding keys. Every fixed hex digit multiplies the work by 16; a full
  8-digit ID (2^32 keys) takes well under a second on a modern NVIDIA GPU.
* **MeshCore**: an Ed25519 identity key whose public key, in hex, starts with
  a prefix of your choice (any length; each character multiplies the work
  by 16).

Every hit is re-derived independently on the host with
[curve25519-dalek](https://github.com/dalek-cryptography/curve25519-dalek)
before it is printed or returned. Private keys are generated from the
operating system's CSPRNG, are never written to disk by this program and are
never logged by the server.

The Meshtastic mode follows the algorithm and pattern syntax of
[vanity.freq51.net](https://vanity.freq51.net) (an in-browser grinder), which
is also the reference the derivation is checked against. Credit to its author
for the idea.

> **No warranty.** This software is provided "as is", without warranty of any
> kind, and you use it at your own risk (see [LICENSE](LICENSE), MIT). Vanity
> keys are ordinary random keys with a chosen public-key property; they are
> neither stronger nor weaker than any other key, but you are responsible for
> keeping them secret and for what you install on your radios.

## Contents

- [How the Meshtastic node ID is derived](#how-the-meshtastic-node-id-is-derived)
- [Install and build](#install-and-build)
- [Command-line usage](#command-line-usage)
- [Applying a key to a radio](#applying-a-key-to-a-radio)
- [API server](#api-server)
- [RunPod Serverless](#runpod-serverless)
- [Performance](#performance)
- [Security](#security)
- [Testing](#testing)
- [How the search works](#how-the-search-works)
- [License](#license)

## How the Meshtastic node ID is derived

**Firmware requirement: Meshtastic 2.8.x or later.** Up to and including
2.7.x the node number comes from the device's hardware MAC address and has
nothing to do with the key, so a vanity key changes nothing there.

Since firmware 2.8 (commit `8267bb22`, "Packet Signing via XEdDSA") the node
number is the CRC-32 of the node's 32-byte X25519 public key:

* `NodeDB::createNewIdentity` sets `myNodeInfo.my_node_num =
  crc32Buffer(public_key, 32)`, and `Router::verifyFirstContactNodeInfo`
  checks the same relation on incoming node info.
* `crc32Buffer` is the standard reflected CRC-32 (IEEE 802.3, as in zlib):
  polynomial `0xEDB88320`, initial value `0xFFFFFFFF`, final XOR
  `0xFFFFFFFF`. Its check value for `"123456789"` is `0xCBF43926`.
* The node ID shown everywhere is the node number as `!%08x`, e.g.
  `!dc805a49`.
* Node numbers below 4 (`NUM_RESERVED`) and `0xFFFFFFFF` (the broadcast
  address) are rejected by the firmware's `pickNewNodeNum`, which then picks
  a random number instead. This tool never returns such a key.

The private key is a clamped 32-byte X25519 secret (`key[0] &= 248; key[31]
&= 127; key[31] |= 64`), the public key is `X25519(secret, 9)`. Both are
exactly what `meshtastic --set security.private_key` and `security.public_key`
hold.

## Install and build

Requirements: a Rust toolchain (stable, 1.80 or newer; edition 2021). The
build script generates the Ed25519 basepoint table and, with the `cuda`
feature, compiles the CUDA kernel with `nvcc`.

```bash
git clone https://github.com/h3lix1/MTVanityKey.git
cd MTVanityKey
```

### CPU only (Linux, macOS, Windows)

```bash
cargo build --release
./target/release/mtvanitykey meshtastic dc80 --cpu-only
```

The CPU workers use the AArch64 CRC32 instructions where available (Apple
silicon, recent ARM servers) and hand-scheduled AArch64 field arithmetic;
x86-64 uses the portable code.

### macOS with Apple Metal (default on Apple silicon)

```bash
cargo build --release
./target/release/mtvanitykey --self-test
./target/release/mtvanitykey meshtastic dc80
```

On macOS the Metal backend is built in and used automatically together with
the CPU workers. The shader is compiled at first use (about a second).

### NVIDIA CUDA

```bash
# CUDA Toolkit 12.x with nvcc on PATH (12.8 or newer for RTX 50-series / sm_120).
cargo build --release --features cuda
./target/release/mtvanitykey --self-test
./target/release/mtvanitykey meshtastic dc80
```

The build script picks the GPU architecture by querying `nvidia-smi` for the
compute capability of GPU 0 (`9.0` becomes `sm_90`). Without `nvidia-smi` (a
build machine without a GPU) it falls back to the oldest architecture the
toolkit supports and embeds PTX, which the driver JIT-compiles forward. Set
the architecture explicitly to be sure:

```bash
CUDA_ARCH=sm_120 cargo build --release --features cuda   # RTX 50xx (Blackwell)
CUDA_ARCH=sm_90  cargo build --release --features cuda   # H100 / GH200
CUDA_ARCH=sm_89  cargo build --release --features cuda   # RTX 40xx
CUDA_ARCH=sm_86  cargo build --release --features cuda   # RTX 30xx
```

`nvcc` is found on `PATH` or through `CUDA_HOME`, `CUDA_PATH` or `CUDA_NVCC`.
At run time the CUDA driver library (`libcuda.so.1` / `nvcuda.dll`) is loaded
dynamically, so the binary starts without a GPU and reports a clear error
when `--gpu-only` is requested. Optional build knobs: `MTVK_CUDA_LB_THREADS`
(block size, default 256), `MTVK_CUDA_BATCH_INV_WINDOW` (candidates per
batched inversion, default 256), `MTVK_CUDA_DEFINES` (extra `nvcc`
arguments).

### Library only, and WebAssembly

The command line, the threaded engine, the API server and the GPU backends
sit behind the `cli` cargo feature, which is on by default. Without it the
crate is the portable core: pattern and prefix parsing, hit verification and
`mtvanitykey::stepper`, a single-threaded search that the caller drives in
slices and seeds with 32 bytes from its own CSPRNG. That core builds for
`wasm32-unknown-unknown`:

```bash
rustup target add wasm32-unknown-unknown
cargo build --lib --release --no-default-features --target wasm32-unknown-unknown
```

```toml
mtvanitykey = { git = "https://github.com/h3lix1/MTVanityKey", default-features = false }
```

```rust
use mtvanitykey::stepper::Stepper;
let mut search = Stepper::parse("meshtastic", "!c0de", seed)?; // seed: [u8; 32] from a CSPRNG
let slice = search.run_windows(1000);           // or run_while(|attempts| ...), run_for(Duration)
println!("{}", slice.to_json());                // {"attempts": N, "hits": [...]}
```

The stepper runs the CPU workers' walks and returns hit objects of the same
shape as the API (`"source": "wasm"` in a wasm build, `"cpu"` otherwise),
each re-derived with curve25519-dalek. One thread of a browser does about
2.3 million keys per second (Apple M1 Max, node 23); on wasm32 the field
multiplication uses 32-bit limb products, since 64x64-bit multiplies there
are library calls.

## Command-line usage

```text
mtvanitykey meshtastic <PATTERN> [--hits N] [--json] [--cpu-only | --gpu-only]
                       [--gpu-backend metal|cuda] [--threads N]
                       [--gpu-batch N] [--gpu-iters N] [--max-attempts N]
mtvanitykey meshcore   <HEXPREFIX> [same options]
mtvanitykey serve      [--bind HOST:PORT] [--token TOKEN] [limits...] [worker options]
mtvanitykey --self-test [--gpu-backend metal|cuda]
```

By default every flow uses all CPU cores plus the platform's GPU (Metal on
macOS, CUDA when built with the feature). `--gpu-only` fails loudly if the
GPU cannot be initialised; without it a missing GPU just means CPU-only.
Progress (attempts, rate, probability of having found a key by now, p50/p95
ETA) is printed to stderr; results go to stdout.

### Meshtastic patterns

A pattern is a comma-separated list of up to 64 targets. Each target is up
to 8 hex digits, case-insensitive, with an optional leading `!`; `*`, `?`,
`.`, `x` and `X` are wildcards (any digit). A target shorter than 8 digits is
a prefix: `dc80` means `!dc80????`. All targets in one list must fix the same
digit positions (`dc80,beef` is fine, `dc80,abc` is not). A full 8-digit
target that is reserved by the firmware (`!00000000`..`!00000003`,
`!ffffffff`) is rejected.

```bash
# A node id starting !dc80 (65536 keys expected; milliseconds on a GPU).
mtvanitykey meshtastic dc80

# Any of three words in the first four digits, three keys, as JSON.
mtvanitykey meshtastic '!dead,beef,cafe' --hits 3 --json

# Wildcards: digit 1-4 = dc80, digit 6 = a.
mtvanitykey meshtastic 'dc80*a'

# A full id. 2^32 keys expected: about half a second on an RTX 5090,
# ~10 s on an M1 Max GPU, ~40 s on its CPU cores.
mtvanitykey meshtastic '!deadbeef' --gpu-only
```

Example output:

```text
pattern dc80: 1 target(s), 4 fixed hex digit(s), 65536 keys expected per hit; stopping after 1 hit(s)
workers: 10 CPU thread(s) + Metal GPU (batch 32768, 1024 iters/thread)
hit 1: node id !dc805a49
  private key:     484a47ae2975b17d04b3b75f03d51eb377ea4fa4ea930da9e0339a1f80d22348
  private base64:  SEpHril1sX0Es7dfA9Ues3fqT6Tqkw2p4DOaH4DSI0g=
  public key:      82050a2d4230fae10ebc9e1d902bae8f30914e9b06a4b24ba8214b554188a347
  public base64:   ggUKLUIw+uEOvJ4dkCuujzCRTpsGpLJLqCFLVUGIo0c=
  command:         meshtastic --set security.private_key "base64:SEpHril1sX0Es7dfA9Ues3fqT6Tqkw2p4DOaH4DSI0g="
  source:          cpu
1 hit(s) after 624640 attempts in 78ms (7990785 keys/sec).
```

(The keys above were generated for this README and have never been used.)

### MeshCore prefixes

A MeshCore prefix is 1 to 64 hex characters of the 32-byte Ed25519 public
key, case-insensitive. MeshCore reserves public keys starting with byte
`0x00` or `0xFF`, so prefixes starting `00` or `ff` are rejected and such
keys are never returned.

The command line accepts any length and shows the expected time as it runs;
every character multiplies the work by 16:

| Prefix | Keys expected | RTX 5090 | Apple M1 Max (GPU + CPU) |
| --- | --- | --- | --- |
| 8 characters | 2^32 (4.3 G) | ~0.5 s | ~9 s |
| 9 characters | 2^36 (69 G) | ~6 s | ~2.5 min |
| 10 characters | 2^40 (1.1 T) | ~1.5 min | ~40 min |
| 11 characters | 2^44 (18 T) | ~25 min | ~10 h |
| 12 characters | 2^48 (281 T) | ~6.5 h | ~7 days |

(RTX 5090 at its measured ~12 G keys/s steady state; a 10-character search
there took 3 min 5 s in one test, an unlucky draw at twice the average.)

These are averages; a single search can take several times as long (see the
p50/p95 in the progress line). The API server limits what others may ask
for with `--max-prefix-chars` and `--max-expected-seconds` (see below).

```bash
mtvanitykey meshcore f8a1
mtvanitykey meshcore DEADBEEF --gpu-only --json
```

```text
prefix F8A1: 4 hex character(s), 65536 keys expected
workers: 4 CPU thread(s)
Public Key:  F8A1AE8B169EF802C518CC5D9576C34BEB0ED5825ED46E224D8293882420B620
Private Key: 40929A26B313A5EE205D78E295C4F9BC2F7D9496DC4460A6CD089E30EBE145589BE7D6823BF9715D87F761083E3C2CF2E899E5E3FE95825C41C94096697E94B8
Source:      cpu
1 hit(s) after 5120 attempts in 4ms (1159322 keys/sec).
```

The private key is MeshCore's 64-byte identity-key layout, printed as 128
upper-case hex characters: the first 32 bytes are the clamped Ed25519 scalar
that produces the public key, the last 32 bytes are the random
signing-nonce half of an expanded Ed25519 secret key (fresh CSPRNG output).

### JSON output

`--json` prints one document at the end:

```json
{
  "hits": [
    {
      "flow": "meshtastic",
      "node_id": "!beef981b",
      "node_num": 3203373083,
      "private_key_hex": "d058eff9...",
      "private_key_base64": "0Fjv+eWo...",
      "public_key_hex": "4422535a...",
      "public_key_base64": "RCJTWsLs...",
      "command": "meshtastic --set security.private_key \"base64:0Fjv+eWo...\"",
      "source": "metal"
    }
  ],
  "attempts": 100663296,
  "cpu_attempts": 0,
  "gpu_attempts": 100663296,
  "elapsed_seconds": 0.292,
  "keys_per_second": 344772491.3,
  "stop_reason": "complete"
}
```

MeshCore hits are `{"flow": "meshcore", "public_key": "...", "private_key":
"...", "source": "..."}`.

## Applying a key to a radio

### Meshtastic

Back up the current identity first: changing the private key changes the
node's public key **and its node ID**. Other nodes will see your radio as a
brand-new node, direct-message encryption with existing contacts breaks until
they learn the new key, and anything tied to the old node ID (favourites,
admin keys on other nodes, MQTT topics) must be updated.

```bash
# Save the current keys (and ideally a full config export).
meshtastic --get security.private_key --get security.public_key
meshtastic --export-config > before-vanity.yaml

# Install the new private key; the firmware derives the public key and the
# node number from it (firmware 2.8 or later).
meshtastic --set security.private_key "base64:SEpHril1sX0Es7dfA9Ues3fqT6Tqkw2p4DOaH4DSI0g="
```

After the reboot `meshtastic --info` shows the new public key and the node ID
you searched for. If the ID does not change, the firmware is older than 2.8
(the node number still comes from the MAC address) or the key did not apply.

### MeshCore

Import the 128-hex-character private key as the device's identity key with
your MeshCore client or firmware tooling (the exact menu depends on the
client; the key format is the one MeshCore uses for identity import/export).
The same caution applies: the public key is the node's identity, so existing
contacts see a new node.

## API server

`mtvanitykey serve` exposes both flows over HTTP with a **FIFO job queue**.
The GPU is a single shared resource, so jobs run strictly one at a time in
submission order; a job waits in the queue until the ones before it finish.

```bash
mtvanitykey serve --bind 127.0.0.1:8080 --token "$(openssl rand -hex 16)"
```

Options (defaults in parentheses): `--bind` (127.0.0.1:8080), `--token`
(none; also `MTVANITYKEY_TOKEN`), `--max-queue` jobs including the running one
(32), `--max-hits` per job (16), `--max-expected-seconds` (300): a job whose
expected duration at the measured rate exceeds this is refused with 413,
`--max-expected-attempts` (explicit key cap), `--max-digits` for patterns (8),
`--max-prefix-chars` for MeshCore (64, i.e. only the time limit applies), `--job-timeout` seconds (600): a
running job is stopped and fails, `--result-ttl` seconds (600): unretrieved
results are deleted, `--max-wait` seconds (60) for `wait=` requests,
`--calibrate` seconds (2) of rate measurement at startup, `--assumed-rate`
keys/s (1e8) used for the time limit until a rate has been measured (set it
near the device's rate when skipping calibration), `--handler-threads`
(8), `--quiet`, plus the worker options (`--cpu-only`, `--gpu-only`,
`--gpu-backend`, `--threads`, `--gpu-batch`, `--gpu-iters`).

All responses are JSON. Errors are `{"error": "message"}` with status 400
(invalid request), 401 (token), 404 (unknown job), 405, 413 (exceeds this
server's limits) or 429 (queue full). When a token is configured every
endpoint except `/v1/health` requires `Authorization: Bearer <token>`.

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/meshtastic?pattern=P[&hits=N][&wait=true\|SECS]` | Enqueue a Meshtastic job |
| `GET` | `/v1/meshcore?prefix=HEX[&wait=true\|SECS]` | Enqueue a MeshCore job |
| `GET` | `/v1/jobs/{id}` | Job status; a finished job's result, once |
| `DELETE` | `/v1/jobs/{id}` | Cancel (dequeue or stop) a job |
| `GET` | `/v1/health` | Backend, measured rate, queue length, limits |

Without `wait`, the enqueue call returns **202** with the job's status
document, including its 1-based `position` in the queue (the running job
counts as position 1) and an `eta_seconds` estimate (expected keys of the
jobs ahead plus its own, divided by the measured rate). With `wait=true` (or
a number of seconds, capped by `--max-wait`) the call blocks until the job
finishes and returns **200** with the result; if the wait runs out it returns
202 with the current status and you poll `/v1/jobs/{id}`.

**Results are handed out exactly once.** The first `GET /v1/jobs/{id}` (or
the `wait` response) that carries a finished job's result deletes the job;
later requests get 404. Unretrieved results are deleted after
`--result-ttl`. Results only ever live in memory.

A job's status is one of `queued`, `running`, `done`, `failed` (time limit
or no worker) or `cancelled`. Cancelling a queued job removes it; cancelling
the running job stops its workers within a launch.

### curl examples

```bash
B=http://127.0.0.1:8080
H='Authorization: Bearer YOURTOKEN'   # omit when the server has no token

curl -s -H "$H" "$B/v1/health"
# {"status":"ok","version":"0.1.0","backend":"metal","device":"Apple M1 Max",
#  "cpu_threads":10,"keys_per_second":5.1e8,"queue_length":0,"running":false,
#  "limits":{"max_queue":32,"max_hits":16,"max_expected_seconds":300.0,...}}

# Blocking: wait for the key (200 when done).
curl -s -H "$H" "$B/v1/meshtastic?pattern=dc80&hits=1&wait=true"
# {"id":"3f1c...","flow":"meshtastic","target":"dc80","hits_wanted":1,"status":"done",
#  "position":null,"attempts":1310720,"elapsed_seconds":0.02,...,
#  "result":{"hits":[{"flow":"meshtastic","node_id":"!dc80...","private_key_hex":"...",
#   "private_key_base64":"...","public_key_hex":"...","public_key_base64":"...",
#   "command":"meshtastic --set security.private_key \"base64:...\"","source":"metal"}],
#   "attempts":1310720,"elapsed_seconds":0.02,"keys_per_second":6.5e7}}

# Non-blocking: enqueue, then poll.
curl -s -H "$H" "$B/v1/meshtastic?pattern=%21dead%2Cbeef%2Ccafe&hits=3"
# {"id":"9a8e52afbb1d7ead","status":"queued","position":2,"eta_seconds":0.4,...}
curl -s -H "$H" "$B/v1/jobs/9a8e52afbb1d7ead"
# {"id":"9a8e52afbb1d7ead","status":"running","attempts":12345678,"keys_per_second":...,"eta_seconds":...}
curl -s -H "$H" "$B/v1/jobs/9a8e52afbb1d7ead"        # once done: the result (then 404)

# MeshCore prefix.
curl -s -H "$H" "$B/v1/meshcore?prefix=f8a1&wait=true"
# {..."status":"done","result":{"hits":[{"flow":"meshcore","public_key":"F8A1...","private_key":"...","source":"cuda"}],...}}

# Cancel.
curl -s -H "$H" -X DELETE "$B/v1/jobs/9a8e52afbb1d7ead"
# {"id":"9a8e52afbb1d7ead","status":"cancelled","was":"running"}

# Refusals.
curl -s "$B/v1/meshtastic?pattern=12g4"
# {"error":"\"g\" is not a hex digit. Use 0-9, a-f, or * for any digit"}   (400)
curl -s "$B/v1/meshcore?prefix=0123456789abcdef"
# {"error":"expected 1.845e19 keys is about 36000000000 s at 5.120e8 keys/s; this server accepts jobs up to 300 s"}   (413)
```

Remember to URL-encode `!` (`%21`), `,` (`%2C`) and `*` (`%2A`) in query
strings when your client does not do it for you.

## RunPod Serverless

`deploy/runpod/` runs the CUDA build as a
[RunPod Serverless](https://docs.runpod.io/serverless/overview) worker: a
queue-based endpoint that scales GPU workers from zero, so an idle endpoint
costs nothing. Each worker starts `mtvanitykey serve` on `127.0.0.1` (GPU
only, quiet) and `handler.py` forwards jobs to it. Results are encrypted to
a public key the client sends with the job, so plaintext private keys never
reach RunPod's job store. The GPU operator can still in principle see the
keys while they are generated; the warning under [Security](#security)
applies.

**Image.** `deploy/runpod/Dockerfile` builds three binaries (`sm_86`,
`sm_89`, `sm_120`; each also embeds PTX for newer GPUs) in
`nvidia/cuda:12.8.1-devel-ubuntu24.04` and copies them into an
`ubuntu:24.04` runtime with Python, `runpod` and `cryptography`. No CUDA
runtime is needed in the image: the NVIDIA container toolkit provides the
host driver's `libcuda` and `nvidia-smi`. The handler picks the highest build
at or below the GPU's compute capability. The workflow
`.github/workflows/runpod-image.yml` publishes it as
`ghcr.io/<owner>/mtvanitykey-runpod` (`latest` from `main`, plus a
`sha-<commit>` tag on every build). To build it yourself:

```bash
docker build -f deploy/runpod/Dockerfile -t mtvanitykey-runpod .
```

**Endpoint.** In the RunPod console create a Serverless endpoint from the
image (make the GHCR package public, or add registry credentials), choose
the GPU types (for example RTX 5090, then RTX 4090 as a fallback), set
active workers to 0 and max workers as you like, and leave the container
start command empty. No environment variables or volumes are needed.
Execution timeout: a few minutes is plenty (the handler stops a job after
300 s).

**Job input** (`POST https://api.runpod.ai/v2/<endpoint id>/run` with
`{"input": ...}` and your RunPod API key):

```json
{"flow": "meshtastic", "pattern": "!dead*", "hits": 1, "client_pubkey": "<base64url P-256 point>"}
{"flow": "meshcore", "prefix": "BEEF", "client_pubkey": "<base64url P-256 point>"}
```

`hits` is 1 to 4 (Meshtastic only). By default patterns are limited to 8
fixed hex digits and 64 targets, prefixes to 9 characters, and jobs to 600 s
of expected search. The limits are set per endpoint with environment
variables (RunPod: Manage → Edit Endpoint → Environment Variables), no image
rebuild needed:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MTVK_MAX_PREFIX_CHARS` | 9 | Longest MeshCore prefix (1 to 64) |
| `MTVK_MAX_DIGITS` | 8 | Most fixed hex digits in a Meshtastic pattern |
| `MTVK_MAX_HITS` | 4 | Most keys per Meshtastic job |
| `MTVK_MAX_EXPECTED_SECONDS` | 600 | Refuse jobs expected to take longer on this GPU |
| `MTVK_JOB_TIMEOUT` | 300 | Stop a running job after this many seconds |
| `MTVK_ASSUMED_RATE` | 5e9 | Keys/s the time limit assumes until the first job measures the GPU |

Keep the endpoint's execution timeout above `MTVK_JOB_TIMEOUT`.
`client_pubkey` is required: an uncompressed P-256 point (65 bytes,
unpadded base64url), fresh for every job. Invalid input fails the job with
the API server's message as `error`.

**Progress.** While the job is `IN_PROGRESS`, `/status/<job id>` shows
`{"phase": "starting"|"running", "gpu": "...", "attempts": N,
"keys_per_second": R, "expected_attempts": E}` as its output, updated about
every 250 ms. Cancelling the RunPod job (`/cancel/<job id>`) stops the search.

**Output.**

```json
{"v": 1, "gpu": "NVIDIA GeForce RTX 5090", "attempts": 123, "elapsed_seconds": 0.5, "keys_per_second": 9.3e9,
 "enc": {"alg": "ECDH-P256+HKDF-SHA256+AES-256-GCM", "epk": "...", "iv": "...", "ct": "..."}}
```

To decrypt: ECDH between the client's private key and `epk` (65-byte raw
point), then `key = HKDF-SHA256(ikm = shared secret, salt = epk ||
client_pubkey, info = "meshkeys.io/v1 result", 32 bytes)`, then AES-256-GCM
with `iv` (12 bytes), additional data `"meshkeys.io/v1"`, and the 16-byte tag
at the end of `ct`. The plaintext is `{"hits": [...]}` with the API's hit
objects. `deploy/runpod/ecies.py` implements both sides; in a browser the
client is a few WebCrypto calls with a non-extractable key.

**Logging.** The handler logs one line per job (outcome, duration and GPU)
and never keys, patterns or results; the RunPod SDK's own logging is set to
warnings (`RUNPOD_LOG_LEVEL`).

**Testing without RunPod**, on any machine with an NVIDIA GPU and the
binaries built (Python 3 with the two packages from `requirements.txt`):

```bash
cd deploy/runpod
python3 selftest.py                                   # encryption and input checks, no GPU
python3 selftest.py input meshtastic '!c0de' > job.json   # fresh client key in client-key.pem
MTVK_BINARY=../../target/release/mtvanitykey RUNPOD_LOG_LEVEL=INFO \
  python3 handler.py --test_input "$(cat job.json)" 2>&1 | tee run.log
python3 selftest.py decrypt client-key.pem < run.log  # decrypts and re-derives every hit
```

Measured on an RTX 5090: the handler is ready about 1.7 s after its process
starts (importing the RunPod SDK, overlapped with `mtvanitykey serve`
start-up and CUDA initialisation, about 0.5 s); RunPod's own container
start comes on top. A job costs about 0.5 s of fixed overhead (mostly the
per-job CUDA context and first launch) plus the search: an 8-digit
Meshtastic pattern took 1.0 s on average (0.6-1.7 s over 20 jobs), an
8-character MeshCore prefix 0.9 s (0.5-2.3 s), four 8-digit hits 2.8 s.

## Performance

Rates measured with this crate (release build, default settings, one job),
in keys per second:

| Machine | Meshtastic (node ID) | MeshCore (prefix) |
| --- | --- | --- |
| NVIDIA RTX 5090 (CUDA, sm_120) | 9.3 G/s | 10.1 G/s (whole 8-character job; ~12 G/s steady state) |
| Apple M1 Max GPU (Metal, 32 cores) | 416 M/s | 425 M/s |
| Apple M1 Max CPU (10 cores) | 98 M/s | 96 M/s |

Metal and CPU rates add up when both run (the default on a Mac). The CUDA
build also runs the CPU workers, but on a GPU this fast they do not matter.
A MeshCore job on the RTX 5090 ends after one hit, i.e. within about a
second, so its whole-job average includes the ~0.3 s of context creation
and first launch; the marginal rate between longer and shorter runs is
about 12 G/s, the same kernel speed as the Meshtastic walk minus the CRC.
The Meshtastic numbers are from runs of 7-10 s (16 hits of a full 8-digit
pattern); the M1 Max numbers from 10 s runs.

The expected number of keys for a pattern with `n` fixed hex digits (or a
MeshCore prefix of `n` characters) is `16^n` (divided by the number of
targets in a Meshtastic list). The expected time to the first hit:

| Fixed digits | Expected keys | RTX 5090 | M1 Max GPU | M1 Max CPU |
| --- | --- | --- | --- | --- |
| 1 | 16 (16^1) | < 1 ms | < 1 ms | < 1 ms |
| 2 | 256 (16^2) | < 1 ms | < 1 ms | < 1 ms |
| 3 | 4,096 (16^3) | < 1 ms | < 1 ms | < 1 ms |
| 4 | 65,536 (16^4) | < 1 ms | < 1 ms | < 1 ms |
| 5 | 1,048,576 (16^5) | < 1 ms | 3 ms | 11 ms |
| 6 | 16,777,216 (16^6) | 2 ms | 40 ms | 171 ms |
| 7 | 268,435,456 (16^7) | 29 ms | 645 ms | 2.7 s |
| 8 | 4,294,967,296 (16^8) | 462 ms | 10.3 s | 43.8 s |

These are search times only. Every job also pays a fixed start-up cost
(about 0.3 s for CUDA context creation and the first launch, 0.5-1 s for
compiling the Metal pipeline), which dominates patterns of six digits or fewer.

A search is memoryless: the time to a hit is exponentially distributed around
these means (the median is 0.69x, and 5 percent of searches take more than
3x the mean). The progress line shows the probability of having found a key
by now and the p50/p95 ETA.

For comparison, in-browser WebAssembly grinders such as vanity.freq51.net
run at a few million keys per second per core, so a full 8-digit node ID
takes on the order of an hour there against well under a second on an RTX
5090 and some ten seconds on an Apple M1 Max.

## Security

* Private keys come from the operating system's CSPRNG (`OsRng`, keying a
  ChaCha20 stream per worker). Choosing a pattern does not weaken a key: the
  pattern only constrains the public key, which is public anyway, and
  recovering the private key from it still means solving the discrete
  logarithm on the full curve. The library's `stepper` takes its 32-byte
  seed from the caller instead (in a browser, `crypto.getRandomValues`) and
  expands it the same way; an all-zero seed or one already used in the
  process is refused.
* Keys are printed to stdout or returned in an HTTP response and otherwise
  live only in memory. Nothing is written to disk; the server logs method,
  path, status code and job statistics, never keys.
* **Anyone who operates an API server sees every key it returns**, and so
  does anyone who can read the traffic. Only use a server you trust; prefer
  running your own (`mtvanitykey serve` on the machine with the GPU). If you
  expose a server beyond `localhost`, put it behind an HTTPS reverse proxy
  (nginx, Caddy, ...), set `--token`, and keep the queue limits.
* The server binds to `127.0.0.1` by default. Its token comparison is
  constant-time; it has no accounts, no persistence and no rate limiting
  beyond the queue size, so do not expose it to the open internet unprotected.
* A vanity key makes your node easy to recognise by design; it does not make
  it easier to impersonate (that would require the private key).

## Testing

```bash
cargo test --release                       # CPU + Metal (on macOS) tests
cargo test --release --features cuda       # + CUDA GPU tests (needs an NVIDIA GPU)
cargo test --release --no-default-features # the portable core alone
cargo build --lib --no-default-features --target wasm32-unknown-unknown
```

What the tests cover:

* **Oracle vectors** (`tests/meshtastic_oracle_vectors.txt`, 200 lines of
  `secret pubkey nodeid`) generated with the reference implementation at
  vanity.freq51.net: the host derivation (dalek X25519 + bitwise CRC-32), both
  CRC-32 implementations (bitwise and slicing-by-4; the AArch64 `crc32x` path
  where present), the CPU Montgomery-u walk and the Metal / CUDA kernels
  (each vector's public key computed on the device, byte for byte, and its
  node id matched as a fully fixed target) all agree with the oracle.
  Setting `MESHTASTIC_ORACLE_JS` to a local copy of the site's `oracle.js`
  (with `node` installed) additionally checks 64 fresh random vectors.
* **Pattern and prefix parsers**: every accepted and rejected form,
  wildcards, lists, reserved IDs and prefixes, the length limits.
* **Field arithmetic**: the CPU 4x64-bit field against `num-bigint` on edge
  values and random expression chains, the AArch64 assembly bit-exact with
  the portable code; the Metal shader's `fe_mul`/`fe_sq`/`fe_invert` against
  the reference ref10 chain and `num-bigint` with a host model proving the
  documented input bounds never overflow; the build script's bignum curve
  arithmetic against dalek; the basepoint and step tables entry by entry.
* **Walks**: every candidate of the CPU y-walk and u-walk equals dalek's
  derivation for `base + 8k` (window boundaries, byte carries, restarts,
  desynchronisation detection and the fallback path).
* **GPU kernels** (Metal, and CUDA when built): planted keys at every window
  boundary of multi-window walks must be reported exactly (no extra, no
  missing), pipelined launches return their own results in order, hit
  counts of short patterns/prefixes agree statistically with `16^-n`, every
  reported hit re-derives, and the pipelined worker loops pair results with
  the right launch at every depth.
* **CUDA without a GPU**: `src/kernels/ed25519.cu` is compiled as host C++
  (`tests/cuda_host/`) and run over reference vectors (field ops vs
  `num-bigint`, full `search_kernel` / `mesh_kernel` launches vs dalek),
  including a tiny-window build that stresses the window bookkeeping. Needs a
  C++ compiler (`CXX`); set `MTVK_ALLOW_NO_CXX=1` to skip.
* **Stepper**: short patterns and prefixes are found single-threaded, every
  hit re-derives and has the API's JSON shape, the stopping rules and the
  per-call hit cap hold, a corrupted walk falls back and still verifies,
  and caller-supplied seeds expand like the workers' (an all-zero or
  repeated seed is refused). The 32-bit-limb field multiplication used on
  wasm32 is bit-exact with the portable one.
* **Statistics**: base scalars are uniform with the documented clamp bits
  (bit 253 free, so every real key is reachable), worker RNG streams are
  independent, node IDs over the walk are uniform.
* **Engine and API**: searches complete, stop, time out and respect attempt
  limits; the HTTP server is started on an ephemeral port and exercised for
  both flows: `wait=true`, concurrent submissions, FIFO positions and ETAs,
  cancel (queued and running), queue-full 429, invalid patterns, limits,
  bearer-token auth, job time limit, result TTL, one-time result retrieval
  and shutdown, with every returned key re-derived with dalek.

## How the search works

Both flows walk the clamped scalars `base, base+8, base+16, ...` (adding 8
preserves the clamp) from CSPRNG-chosen bases. On the GPU each thread does
one fixed-base scalar multiplication with a 43x32-entry basepoint table,
then derives candidates in windows of 256 around a window middle `M` from an
affine table of `i*S` (`S = 8B`): with `A = X*x_i`, `B = Y*y_i`,
`K = (XY)(d x_i y_i)` the complete twisted-Edwards addition law gives
`y(M +- iS)` from the same three products, and one batched field inversion
per window turns the projective values affine. That is about 5.5 field
multiplications per candidate.

* MeshCore compares the canonical `y` encoding's low bytes with the prefix
  (the Ed25519 public key is `y` with the sign of `x` in the top bit; that
  bit only matters for a full 64-character prefix, which the host checks).
* Meshtastic needs the X25519 public key, the Montgomery `u = (1+y)/(1-y)`.
  Writing `y = n/d` gives `u = (d+n)/(d-n)`, so the batch inversion simply
  runs over `d-n` instead of `d`: the conversion costs nothing. The 32-byte
  `u` is CRC-32'd with slicing-by-4 tables held in shared/threadgroup
  memory, the masked node number is tested against a 4096-bit bitmap and
  then exactly against up to 64 targets.

The CPU workers run the same walk with a y-only (or u-only) differential
addition in a hand-scheduled 4x64-bit field (~6 multiplications and 2
squarings per candidate), checking themselves against dalek at every restart
and periodically. Every hit anywhere is re-derived with curve25519-dalek and
a bitwise CRC-32 before it is reported, so an arithmetic fault could only
ever miss keys, never report a wrong one.

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 MTVanityKey contributors.
No warranty; use at your own risk.
