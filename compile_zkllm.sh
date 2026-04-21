#!/bin/bash
# compile_zkllm.sh
# Runs at container start. Skips compilation entirely if all 6 binaries already exist.

set -euo pipefail

REPO="${ZKLLM_REPO:-/app/zkllm-ccs2024}"
cd "$REPO"

BINARIES=(ppgen commit-param rmsnorm self-attn ffn skip-connection)

# ── Skip if already compiled ──────────────────────────────────────────────────
ALL_EXIST=1
for bin in "${BINARIES[@]}"; do
    if [[ ! -x "$REPO/$bin" ]]; then
        ALL_EXIST=0
        break
    fi
done

if [[ $ALL_EXIST -eq 1 ]]; then
    echo "[compile] All binaries already exist, skipping compilation."
    exit 0
fi

echo "[compile] Working in: $REPO"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[FATAL] nvidia-smi not found. This container requires a GPU host." >&2
    exit 1
fi

CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
    | head -n1 | tr -d '.' | tr -d ' ')
GPU_ARCH="sm_${CAP}"
echo "[compile] Detected GPU architecture: ${GPU_ARCH}"

sed -i 's|NVCC := $(CONDA_PREFIX)/bin/nvcc|NVCC := /usr/local/cuda/bin/nvcc|' Makefile
sed -i 's|INCLUDES := -I$(CONDA_PREFIX)/include|INCLUDES := -I/usr/local/cuda/include|' Makefile
sed -i 's|LIBS := -L$(CONDA_PREFIX)/lib|LIBS := -L/usr/local/cuda/lib64|' Makefile
sed -i "s|ARCH := sm_[0-9]*|ARCH := ${GPU_ARCH}|" Makefile

echo "[compile] Makefile patched for ${GPU_ARCH}"

NCPU=$(nproc)
echo "[compile] Building with -j${NCPU} ..."
make clean
make -j"${NCPU}" ppgen commit-param rmsnorm self-attn ffn skip-connection 2>&1 | tail -20

MISSING=0
for bin in "${BINARIES[@]}"; do
    if [[ -x "$REPO/$bin" ]]; then
        echo "[compile] OK: $bin"
    else
        echo "[FATAL] Missing binary: $bin" >&2
        MISSING=1
    fi
done

if [[ $MISSING -ne 0 ]]; then
    echo "[FATAL] One or more binaries failed to build." >&2
    exit 1
fi

echo "[compile] All binaries built successfully. GPU arch: ${GPU_ARCH}"
