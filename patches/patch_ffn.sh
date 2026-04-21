#!/bin/bash
# patch_ffn.sh — shrink the SwiGLU lookup table range in ffn.cu
# Shrinks from 2^22 to 2^20 so tLookup divides D = SEQ_LEN * INTER = 512*2048 = 2^20 evenly.
#
# Idempotent: safe to run multiple times.
# Usage: bash patch_ffn.sh /path/to/zkllm-ccs2024/ffn.cu

set -euo pipefail

FFN_FILE="${1:?Usage: $0 <path-to-ffn.cu>}"

if grep -q 'swiglu(-(1 << 19), 1 << 20' "$FFN_FILE"; then
    echo "[INFO] ffn.cu already patched, skipping."
    exit 0
fi

sed -i -E \
    's/tLookupRangeMapping swiglu\(-\(1 << [0-9]+\), 1 << [0-9]+, swiglu_values\);/tLookupRangeMapping swiglu(-(1 << 19), 1 << 20, swiglu_values);/' \
    "$FFN_FILE"

if ! grep -q 'swiglu(-(1 << 19), 1 << 20' "$FFN_FILE"; then
    echo "[FATAL] ffn.cu patch regex did not match — swiglu declaration not found." >&2
    exit 1
fi

echo "[INFO] ffn.cu patched (swiglu table -> 2^20)."
