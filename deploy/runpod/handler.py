#!/usr/bin/env python3
"""RunPod Serverless worker for MTVanityKey.

On start: pick the CUDA build that matches GPU 0 (the highest of sm_86,
sm_89 and sm_120 at or below its compute capability; each binary also
carries PTX, so newer GPUs JIT it), launch `mtvanitykey serve` on 127.0.0.1
with the GPU only, wait for `/v1/health`, then hand the loop to
`runpod.serverless.start`.

Per job (input `{"flow": "meshtastic", "pattern": "...", "hits": 1-4,
"client_pubkey": "..."}` or `{"flow": "meshcore", "prefix": "...",
"client_pubkey": "..."}`): refuse it without a valid `client_pubkey`,
enqueue it on the local API (which validates the pattern and the limits;
its errors come back as `{"error": "..."}`), report progress about every
250 ms, and return the result encrypted to `client_pubkey` (see ecies.py),
so plaintext private keys never reach RunPod's job store. A cancelled or
interrupted job is cancelled on the local server too.

Nothing here logs keys, patterns, inputs or outputs: one line per job
("job ok", the duration and the GPU), and the RunPod SDK's own logging is
turned down to warnings unless RUNPOD_LOG_LEVEL says otherwise.

Environment: MTVK_BIN_DIR (directory of mtvanitykey-sm_*; default
/opt/mtvanitykey/bin), MTVK_BINARY (use this binary instead), MTVK_PORT
(default 8080), and the job limits below (MTVK_MAX_*, MTVK_JOB_TIMEOUT).
"""

import time

T0 = time.monotonic()

import asyncio  # noqa: E402
import atexit  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import urllib.error  # noqa: E402
import urllib.parse  # noqa: E402
import urllib.request  # noqa: E402

# The SDK logs job inputs and outputs at DEBUG, its default level.
os.environ.setdefault("RUNPOD_LOG_LEVEL", "WARN")

import ecies  # noqa: E402

# Imported in `main` while the server initialises CUDA (the import alone
# takes over a second); `selftest.py` imports this module without it.
runpod = None

BIN_DIR = os.environ.get("MTVK_BIN_DIR", "/opt/mtvanitykey/bin")
# (binary suffix, minimum compute capability), highest first.
ARCHES = (("sm_120", (12, 0)), ("sm_89", (8, 9)), ("sm_86", (8, 6)))
PORT = int(os.environ.get("MTVK_PORT", "8080"))
BASE = f"http://127.0.0.1:{PORT}"
# The local job is polled often (a short job ends within a poll); progress
# goes to RunPod at most every PROGRESS_SECONDS.
POLL_SECONDS = 0.05
PROGRESS_SECONDS = 0.25
READY_TIMEOUT = 180.0
# Job limits, configurable per endpoint through environment variables
# (defaults in parentheses): MTVK_MAX_PREFIX_CHARS (9), MTVK_MAX_DIGITS (8),
# MTVK_MAX_HITS (4), MTVK_MAX_EXPECTED_SECONDS (600; jobs expected to take
# longer at the measured rate are refused), MTVK_JOB_TIMEOUT (300 s) and
# MTVK_ASSUMED_RATE (5e9 keys/s: the rate the time limit uses until the
# first job has measured the real one; the server skips calibration to
# start fast, and a CPU-scale guess would refuse long GPU jobs).
# Keep the endpoint's own execution timeout above the job timeout.
LIMITS = {
    "--max-prefix-chars": os.environ.get("MTVK_MAX_PREFIX_CHARS", "9"),
    "--max-digits": os.environ.get("MTVK_MAX_DIGITS", "8"),
    "--max-hits": os.environ.get("MTVK_MAX_HITS", "4"),
    "--max-expected-seconds": os.environ.get("MTVK_MAX_EXPECTED_SECONDS", "600"),
    "--job-timeout": os.environ.get("MTVK_JOB_TIMEOUT", "300"),
    "--assumed-rate": os.environ.get("MTVK_ASSUMED_RATE", "5e9"),
}
# MTVK_REFRESH_WORKER (0): ask RunPod to reset the worker after every job.
# Off by default: RunPod answers the reset by starting a fresh container on
# that worker a minute or so later, which idles (and is billed) before it
# stops. A short endpoint idle timeout stops workers sooner and cheaper.
REFRESH_WORKER = os.environ.get("MTVK_REFRESH_WORKER", "0") == "1"

# Fanned-out searches (several jobs for one request, the first key wins):
# a job may carry a `group_url` under MTVK_GROUP_URL_PREFIX (unset: the
# feature is off and the field is ignored). While searching, the handler
# polls it every GROUP_POLL_SECONDS; once it answers {"done": true} the job
# ends normally without keys ({"superseded": true}). Ending normally, rather
# than being cancelled by RunPod, lets refresh_worker stop the worker
# cleanly: a cancelled job's result is rejected and the worker then exits
# in a way RunPod treats as a failure (restarts and retries, billed).
GROUP_URL_PREFIX = os.environ.get("MTVK_GROUP_URL_PREFIX", "")
GROUP_POLL_SECONDS = 2.0

# One job at a time per worker.
SERVE_ARGS = [
    "--gpu-only", "--quiet", "--calibrate", "0",
    *(arg for pair in LIMITS.items() for arg in pair),
    "--max-queue", "4", "--result-ttl", "60", "--max-wait", "0",
    "--handler-threads", "4",
]

serve = None
gpu_name = "unknown GPU"


def log(message):
    print(message, file=sys.stderr, flush=True)


def gpu_info():
    """((major, minor), name) of GPU 0 from nvidia-smi."""
    out = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=compute_cap,name", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=60, check=True,
    ).stdout
    cap, name = (field.strip() for field in out.splitlines()[0].split(",", 1))
    major, minor = (int(part) for part in cap.split("."))
    return (major, minor), name


def pick_binary(cap):
    if os.environ.get("MTVK_BINARY"):
        return os.environ["MTVK_BINARY"]
    for arch, minimum in ARCHES:
        path = os.path.join(BIN_DIR, f"mtvanitykey-{arch}")
        if cap >= minimum and os.access(path, os.X_OK):
            return path
    raise SystemExit(f"no mtvanitykey build for compute capability {cap[0]}.{cap[1]} in {BIN_DIR}")


def http(method, path, timeout=10.0):
    """(status, JSON body) of a request to the local server."""
    request = urllib.request.Request(BASE + path, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.loads(error.read())
        except ValueError:
            return error.code, {"error": f"HTTP {error.code}"}


def stop_serve():
    if serve is not None and serve.poll() is None:
        serve.terminate()
        try:
            serve.wait(timeout=5)
        except subprocess.TimeoutExpired:
            serve.kill()


def start_serve(while_starting=None):
    """Launch `mtvanitykey serve`, run `while_starting()` (if any) while it
    initialises, and wait until it answers /v1/health."""
    global serve, gpu_name
    cap, name = gpu_info()
    binary = pick_binary(cap)
    serve = subprocess.Popen(
        [binary, "serve", "--bind", f"127.0.0.1:{PORT}", *SERVE_ARGS],
        stdin=subprocess.DEVNULL,
    )
    atexit.register(stop_serve)
    if while_starting is not None:
        while_starting()
    deadline = time.monotonic() + READY_TIMEOUT
    while True:
        if serve.poll() is not None:
            raise SystemExit(f"mtvanitykey serve exited with status {serve.returncode}")
        try:
            status, health = http("GET", "/v1/health", timeout=1.0)
            if status == 200:
                break
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise SystemExit("mtvanitykey serve did not become ready")
        time.sleep(0.05)
    gpu_name = health.get("device") or name
    log(f"ready in {time.monotonic() - T0:.2f} s: {gpu_name} (compute {cap[0]}.{cap[1]}, {os.path.basename(binary)})")


def group_url(job_input):
    """The job's group_url if it is allowed (MTVK_GROUP_URL_PREFIX), else None."""
    url = job_input.get("group_url") if isinstance(job_input, dict) else None
    if not GROUP_URL_PREFIX or not isinstance(url, str) or len(url) > 512:
        return None
    if not url.startswith(GROUP_URL_PREFIX) or any(c.isspace() for c in url):
        return None
    return url


def group_done(url):
    """True if the group says the search is over (failures count as not)."""
    request = urllib.request.Request(url, headers={"User-Agent": "mtvanitykey-runpod"})
    try:
        with urllib.request.urlopen(request, timeout=3.0) as response:
            return json.loads(response.read(4096)).get("done") is True
    except (OSError, ValueError, AttributeError):
        return False


def local_path(job_input):
    """Validate the job input; return (local API path, client public key)."""
    if not isinstance(job_input, dict):
        raise ValueError("input must be a JSON object")
    if job_input.get("client_pubkey") is None:
        raise ValueError("client_pubkey is required: results are only returned encrypted")
    client = ecies.parse_client_pubkey(job_input["client_pubkey"])
    flow = job_input.get("flow")
    if flow == "meshtastic":
        pattern = job_input.get("pattern")
        hits = job_input.get("hits", 1)
        if not isinstance(pattern, str):
            raise ValueError("pattern must be a string")
        if len(pattern) > 4096:
            raise ValueError("pattern is too long")
        if isinstance(hits, bool) or not isinstance(hits, int):
            raise ValueError("hits must be a whole number from 1 to 4")
        query = {"pattern": pattern, "hits": str(hits)}
    elif flow == "meshcore":
        prefix = job_input.get("prefix")
        if not isinstance(prefix, str):
            raise ValueError("prefix must be a string")
        if len(prefix) > 4096:
            raise ValueError("prefix is too long")
        query = {"prefix": prefix}
    else:
        raise ValueError('flow must be "meshtastic" or "meshcore"')
    return f"/v1/{flow}?{urllib.parse.urlencode(query)}", client


# The SDK's progress_update posts from a detached thread, so an update can
# reach RunPod after the job's result and flip it back to IN_PROGRESS for
# good. Updates are sent from threads kept here instead, and the handler
# waits for the last one before it returns the result.
progress_thread = None


def send_progress(job, status):
    global progress_thread
    # Outside a RunPod worker (local test mode) there is nowhere to send it.
    if runpod is None or not os.environ.get("RUNPOD_WEBHOOK_POST_OUTPUT"):
        return
    if progress_thread is not None and progress_thread.is_alive():
        return  # the previous update is still in flight; skip this one
    attempts = status.get("attempts") or 0
    progress = {
        "phase": "running" if status.get("status") == "running" and attempts > 0 else "starting",
        "gpu": gpu_name,
        "attempts": attempts,
        "keys_per_second": status.get("keys_per_second") or 0.0,
        "expected_attempts": status.get("expected_attempts") or 0.0,
    }
    from runpod.serverless.modules import rp_progress

    progress_thread = threading.Thread(target=rp_progress._thread_target, args=(job, progress), daemon=True)
    progress_thread.start()


def wait_for_progress():
    """Let an in-flight progress update land before the result is sent."""
    if progress_thread is not None:
        progress_thread.join(timeout=10)


async def handler(job):
    started = time.monotonic()
    try:
        path, client = local_path(job.get("input"))
    except ValueError as error:
        log("job refused")
        return {"error": str(error)}

    try:
        status, body = await asyncio.to_thread(http, "GET", path)
    except OSError:
        log("job failed: local engine unavailable")
        return {"error": "the GPU engine is unavailable; retry", "refresh_worker": True}
    if status not in (200, 202):
        log("job refused")
        return {"error": body.get("error", f"HTTP {status}")}

    local_id = body["id"]
    finished = False
    next_progress = time.monotonic()
    group = group_url(job.get("input"))
    next_group = time.monotonic() + GROUP_POLL_SECONDS
    try:
        while True:
            try:
                status, body = await asyncio.to_thread(http, "GET", f"/v1/jobs/{local_id}")
            except OSError:
                log("job failed: local engine unavailable")
                return {"error": "the GPU engine is unavailable; retry", "refresh_worker": True}
            if status != 200:
                finished = True
                await asyncio.to_thread(wait_for_progress)
                log("job failed")
                return {"error": body.get("error", f"HTTP {status}")}
            state = body.get("status")
            if state in ("queued", "running") and group and time.monotonic() >= next_group:
                next_group = time.monotonic() + GROUP_POLL_SECONDS
                if await asyncio.to_thread(group_done, group):
                    # Another job of this search found the key (or it was
                    # stopped): end normally, without keys.
                    try:
                        http("DELETE", f"/v1/jobs/{local_id}", timeout=2.0)
                    except OSError:
                        pass
                    finished = True
                    await asyncio.to_thread(wait_for_progress)
                    log(f"job superseded {time.monotonic() - started:.2f} s {gpu_name}")
                    return {"v": 1, "superseded": True, "gpu": gpu_name, "attempts": body.get("attempts") or 0}
            if state in ("queued", "running"):
                if time.monotonic() >= next_progress:
                    send_progress(job, body)
                    next_progress = time.monotonic() + PROGRESS_SECONDS
                await asyncio.sleep(POLL_SECONDS)
                continue
            finished = True
            await asyncio.to_thread(wait_for_progress)
            if state != "done":
                log(f"job {state}")
                return {"error": body.get("error", state)}
            result = body["result"]
            plaintext = json.dumps({"hits": result["hits"]}, separators=(",", ":")).encode()
            enc = ecies.encrypt(client, plaintext)
            log(f"job ok {time.monotonic() - started:.2f} s {gpu_name}")
            return {
                "v": 1,
                "gpu": gpu_name,
                "attempts": result["attempts"],
                "elapsed_seconds": result["elapsed_seconds"],
                "keys_per_second": result["keys_per_second"],
                "enc": enc,
            }
    finally:
        if not finished:
            # Cancelled by RunPod (task cancellation) or interrupted: stop the
            # local search too. Synchronous: this may run during cancellation.
            try:
                http("DELETE", f"/v1/jobs/{local_id}", timeout=2.0)
            except OSError:
                pass



def import_runpod():
    global runpod
    import runpod as sdk

    runpod = sdk


if __name__ == "__main__":
    start_serve(while_starting=import_runpod)
    runpod.serverless.start({"handler": handler, "refresh_worker": REFRESH_WORKER})
