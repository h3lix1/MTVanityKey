# RunPod Serverless worker image for MTVanityKey (see deploy/runpod/handler.py).
# Build from the repository root:
#   docker build -f deploy/runpod/Dockerfile -t mtvanitykey-runpod .

# --- build: three CUDA builds of the same source -----------------------------
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --profile minimal --default-toolchain stable
WORKDIR /src
COPY Cargo.toml Cargo.lock build.rs ./
COPY src ./src
COPY tests ./tests
# One binary per GPU generation: Ampere (sm_86), Ada (sm_89), Blackwell
# (sm_120). Each embeds SASS for its architecture plus PTX, which newer GPUs
# JIT-compile. Only the build script and the crate rebuild between them.
RUN set -eux; \
    for arch in sm_86 sm_89 sm_120; do \
      CUDA_ARCH=$arch cargo build --release --locked --features cuda; \
      install -D -m 755 target/release/mtvanitykey /out/mtvanitykey-$arch; \
    done

# --- runtime: no CUDA runtime needed (libcuda comes from the host driver) ----
FROM ubuntu:24.04
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*
COPY deploy/runpod/requirements.txt /opt/mtvanitykey/requirements.txt
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r /opt/mtvanitykey/requirements.txt
COPY --from=build /out/ /opt/mtvanitykey/bin/
COPY deploy/runpod/handler.py deploy/runpod/ecies.py deploy/runpod/selftest.py /opt/mtvanitykey/
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RUNPOD_LOG_LEVEL=WARN
WORKDIR /opt/mtvanitykey
RUN python3 selftest.py
CMD ["python3", "handler.py"]
