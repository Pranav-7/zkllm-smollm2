#!/usr/bin/env python3
"""
entrypoint.py — SmolLM2 zkLLM API Server  (v3 — hardened)

Endpoints
---------
GET  /health                     — status, GPU info, disk/memory stats
POST /prove                      — submit query, returns proof_id immediately
GET  /prove/{proof_id}           — poll status: queued → running → done / error
GET  /proof/{proof_id}/download  — download zip of all proof artifacts
POST /verify/{proof_id}          — re-run proof, compare layer hashes
POST /benchmark                  — start benchmark job (returns job_id)
GET  /benchmark/{job_id}         — poll benchmark
"""

import csv, hashlib, json, math, os, shutil, subprocess
import sys, threading, time, uuid, zipfile
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import numpy as np
import psutil
import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── SCRIPT_DIR must be resolved before any os.chdir ──────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import fileio_utils

# ─────────────────────────────────────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CARD = os.environ.get("MODEL_CARD", "HuggingFaceTB/SmolLM2-135M")
SEQ_LEN    = int(os.environ.get("SEQ_LEN",   "512"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
WORKDIR    = os.environ.get("WORKDIR",    "/app/zkllm-workdir")
REPO       = os.environ.get("ZKLLM_REPO", "/app/zkllm-ccs2024")
HF_TOKEN   = os.environ.get("HF_TOKEN",   None)
PROOFS_DIR = os.environ.get("PROOFS_DIR", "/app/proofs")
BENCH_DIR  = os.environ.get("BENCH_DIR",  "/app/benchmarks")
HOST       = os.environ.get("HOST",       "0.0.0.0")
PORT       = int(os.environ.get("PORT",   "8000"))

# Minimum free disk (GB) required before starting a proof or benchmark.
# Each run needs ~500 MB peak. We require 2 GB headroom to be safe.
MIN_FREE_DISK_GB = float(os.environ.get("MIN_FREE_DISK_GB", "2.0"))

for _d in [OUTPUT_DIR, WORKDIR, PROOFS_DIR, BENCH_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_TRUE   = 576
N_HEADS_TRUE  = 9
KV_HEADS_TRUE = 3
HEAD_DIM      = 64
N_GROUPS      = N_HEADS_TRUE // KV_HEADS_TRUE   # 3

HIDDEN  = 1024   # next pow2 >= 576
N_HEADS = 16     # next pow2 >= 9  →  head_dim = 1024/16 = 64 ✓
INTER   = 2048   # next pow2 >= 1536

LOG_SF      = 16
SCALE       = 1 << LOG_SF
LOG_OFF     = 5
VALUE_LOGSF = 16
ACCU_LOGSF  = 20

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────
_model      = None
_tokenizer  = None
_cfg        = None
_n_layers   = None
_gpu_name   = "unknown"
_startup_ok = False
_startup_error: Optional[str] = None

_proof_lock    = threading.Lock()
prove_jobs:     Dict[str, Any] = {}
benchmark_jobs: Dict[str, Any] = {}

# ─────────────────────────────────────────────────────────────────────────────
# GUARDS  — called before every expensive operation
# ─────────────────────────────────────────────────────────────────────────────

def _free_disk_gb(path: str) -> float:
    return shutil.disk_usage(path).free / 1e9

def _free_gpu_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / 1e9

def _free_ram_gb() -> float:
    return psutil.virtual_memory().available / 1e9

def _check_disk(path: str, required_gb: float = MIN_FREE_DISK_GB, label: str = ""):
    free = _free_disk_gb(path)
    if free < required_gb:
        raise RuntimeError(
            f"Not enough disk space{' for ' + label if label else ''}. "
            f"Free: {free:.1f} GB, Required: {required_gb:.1f} GB. "
            f"Path: {path}"
        )

def _check_gpu(required_gb: float = 0.5, label: str = ""):
    free = _free_gpu_gb()
    if free < required_gb:
        raise RuntimeError(
            f"Not enough GPU memory{' for ' + label if label else ''}. "
            f"Free: {free:.2f} GB, Required: {required_gb:.1f} GB. "
            f"Run /health to see current GPU usage."
        )

def _check_ram(required_gb: float = 2.0, label: str = ""):
    free = _free_ram_gb()
    if free < required_gb:
        raise RuntimeError(
            f"Not enough RAM{' for ' + label if label else ''}. "
            f"Free: {free:.1f} GB, Required: {required_gb:.1f} GB."
        )

# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT PADDING
# ─────────────────────────────────────────────────────────────────────────────

def _pad_2d(W, rows, cols):
    P = torch.zeros(rows, cols, dtype=W.dtype)
    P[:W.shape[0], :W.shape[1]] = W
    return P

def _pad_1d(v, n):
    P = torch.zeros(n, dtype=v.dtype)
    P[:v.shape[0]] = v
    return P

def _expand_and_pad_kv(weight):
    w = weight.view(KV_HEADS_TRUE, HEAD_DIM, HIDDEN_TRUE)
    w = w.repeat_interleave(N_GROUPS, dim=0)
    w = w.reshape(N_HEADS_TRUE * HEAD_DIM, HIDDEN_TRUE)
    return _pad_2d(w, N_HEADS * HEAD_DIM, HIDDEN)

def _build_weights(layer_idx):
    layer = _model.model.layers[layer_idx]
    w = {}
    for name, p in layer.named_parameters():
        p = p.detach().float()
        if   'k_proj.weight' in name or 'v_proj.weight' in name:
            w[name] = _expand_and_pad_kv(p)
        elif 'q_proj.weight' in name or 'o_proj.weight' in name:
            w[name] = _pad_2d(p, HIDDEN, HIDDEN)
        elif 'gate_proj.weight' in name or 'up_proj.weight' in name:
            w[name] = _pad_2d(p, INTER, HIDDEN)
        elif 'down_proj.weight' in name:
            w[name] = _pad_2d(p, HIDDEN, INTER)
        elif 'layernorm.weight' in name:
            w[name] = _pad_1d(p, HIDDEN)
    return w

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_pp():
    print("[server] Checking public parameters ...", flush=True)
    ref = _build_weights(0)
    t0  = time.time()
    for name, w in ref.items():
        pp_path = f"{WORKDIR}/{name}-pp.bin"
        if os.path.exists(pp_path):
            continue
        _check_disk(WORKDIR, 0.1, f"ppgen {name}")
        pp_size = (w.shape[0] << LOG_OFF) if w.ndim == 2 else w.shape[0]
        ret = os.system(f"{REPO}/ppgen {pp_size} {pp_path} > /dev/null 2>&1")
        if ret != 0 or not os.path.exists(pp_path):
            raise RuntimeError(f"ppgen failed for {name}")
        print(f"  [pp] generated {name}", flush=True)
    print(f"[server] PP ready in {time.time()-t0:.1f}s", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# COMMIT
# ─────────────────────────────────────────────────────────────────────────────

def _commit(weights, prefix, run_dir):
    """
    Commit layer weights. Deletes int.bin files immediately after commit
    to keep disk usage low (~28 MB peak per layer instead of 840 MB total).
    """
    t0 = time.time()
    total_bytes = 0
    for name, tensor in weights.items():
        w_orig = tensor.float().T if tensor.ndim == 2 else tensor.float()
        w_int  = torch.round(w_orig * SCALE).to(torch.int32)

        pp_path     = f"{WORKDIR}/{name}-pp.bin"           # shared, never deleted
        int_path    = f"{run_dir}/{prefix}-{name}-int.bin"
        commit_path = f"{run_dir}/{prefix}-{name}-commitment.bin"

        # Write int file, run commit, then immediately delete int file
        w_int.cpu().numpy().astype(np.int32).tofile(int_path)
        try:
            if w_int.ndim == 2:
                M, N = w_int.shape
                os.system(
                    f"{REPO}/commit-param {pp_path} {int_path} "
                    f"{commit_path} {M} {N} > /dev/null 2>&1"
                )
            else:
                os.system(
                    f"{REPO}/commit-param {pp_path} {int_path} "
                    f"{commit_path} {w_int.shape[0]} 1 > /dev/null 2>&1"
                )
        finally:
            # Always delete int file — it is large and not needed after commit
            if os.path.exists(int_path):
                os.remove(int_path)

        if os.path.exists(commit_path):
            total_bytes += os.path.getsize(commit_path)
        else:
            raise RuntimeError(f"commit-param produced no output for {name}")

    return time.time() - t0, total_bytes

# ─────────────────────────────────────────────────────────────────────────────
# PROOF PIPELINE  (one layer)
# ─────────────────────────────────────────────────────────────────────────────

def _run_cmd(cmd, tag):
    """Run a shell command; raise RuntimeError with full context on failure."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"[{tag}] command failed (exit {r.returncode})\n"
            f"CMD   : {cmd}\n"
            f"STDOUT: {r.stdout.strip()}\n"
            f"STDERR: {r.stderr.strip()}"
        )

def _prove_layer(layer_idx, input_file, output_file, run_dir):
    prefix   = f"layer-{layer_idx}"
    rn1_out  = f"{run_dir}/{prefix}-rn1-out.bin"
    attn_out = f"{run_dir}/{prefix}-attn-out.bin"
    skip1    = f"{run_dir}/{prefix}-skip1.bin"
    rn2_out  = f"{run_dir}/{prefix}-rn2-out.bin"
    ffn_out  = f"{run_dir}/{prefix}-ffn-out.bin"

    # Intermediate files created during this layer — deleted after skip2
    intermediates = [rn1_out, attn_out, skip1, rn2_out, ffn_out]

    t0 = time.time()

    # STEP 1: input RMSNorm
    X = torch.tensor(
        np.fromfile(input_file, dtype=np.int32).reshape(SEQ_LEN, HIDDEN),
        device='cuda', dtype=torch.float64
    ) / (1 << LOG_SF)
    rms_inv = 1 / torch.sqrt((X ** 2).mean(dim=-1) + _cfg.rms_norm_eps)
    fileio_utils.save_int(rms_inv, 1 << 16, 'rms_inv_temp.bin')
    _run_cmd(
        f"{REPO}/rmsnorm input {input_file} {SEQ_LEN} {HIDDEN}"
        f" {run_dir} {prefix} {rn1_out}",
        f"L{layer_idx}/rms1"
    )
    os.system('rm -f ./rms_inv_temp.bin ./temp*.bin')

    # STEP 2: self-attention
    _run_cmd(
        f"{REPO}/self-attn linear {rn1_out} {SEQ_LEN} {HIDDEN}"
        f" {run_dir} {prefix} {attn_out} {N_HEADS}",
        f"L{layer_idx}/attn-linear"
    )
    Q_int = np.fromfile('temp_Q.bin', dtype=np.int32).reshape(SEQ_LEN, HIDDEN)
    K_int = np.fromfile('temp_K.bin', dtype=np.int32).reshape(SEQ_LEN, HIDDEN)
    V_int = np.fromfile('temp_V.bin', dtype=np.int32).reshape(SEQ_LEN, HIDDEN)

    Q = fileio_utils.to_float(torch.tensor(Q_int, device='cuda'), VALUE_LOGSF)
    K = fileio_utils.to_float(torch.tensor(K_int, device='cuda'), VALUE_LOGSF)
    V = fileio_utils.to_float(torch.tensor(V_int, device='cuda'), VALUE_LOGSF)
    Q = Q.view(SEQ_LEN, N_HEADS, HEAD_DIM).transpose(0, 1).contiguous()
    K = K.view(SEQ_LEN, N_HEADS, HEAD_DIM).transpose(0, 1).contiguous()
    V = V.view(SEQ_LEN, N_HEADS, HEAD_DIM).transpose(0, 1).contiguous()

    inv_freq = 1.0 / (10000 ** (
        torch.arange(0, HEAD_DIM, 2, dtype=torch.float64, device='cuda') / HEAD_DIM
    ))
    freqs = torch.outer(
        torch.arange(SEQ_LEN, dtype=torch.float64, device='cuda'), inv_freq
    )
    emb     = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos().to(Q.dtype), emb.sin().to(Q.dtype)

    def _rot(x):
        a, b = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-b, a), dim=-1)

    Q = Q * cos + _rot(Q) * sin
    K = K * cos + _rot(K) * sin

    import math as _math
    A_   = Q @ K.transpose(-2, -1)
    A    = fileio_utils.to_int64(A_, VALUE_LOGSF)
    mask = torch.triu(
        torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool, device='cuda'), diagonal=1
    )
    A -= torch.max(A * ~mask, dim=-1, keepdim=True).values
    shift = _math.sqrt(HEAD_DIM) * torch.log(
        (torch.exp(
            fileio_utils.to_float(A, ACCU_LOGSF, torch.float64) / _math.sqrt(HEAD_DIM)
        ) * ~mask).sum(dim=-1, keepdim=True).clamp(min=1e-9)
    )
    A -= fileio_utils.to_int64(shift, ACCU_LOGSF)
    attn_w   = torch.exp(
        fileio_utils.to_float(A, ACCU_LOGSF, torch.float64) / _math.sqrt(HEAD_DIM)
    ) * ~mask
    attn_pre = fileio_utils.fromto_int64(attn_w @ V.to(attn_w.dtype), VALUE_LOGSF)
    attn_pre = attn_pre.transpose(0, 1).contiguous().view(SEQ_LEN, HIDDEN)
    fileio_utils.save_int(attn_pre, 1 << VALUE_LOGSF, 'temp_attn_out.bin')

    # Free attention tensors before the next binary call
    del Q, K, V, A_, A, attn_w, attn_pre, mask, shift, emb, cos, sin
    torch.cuda.empty_cache()

    _run_cmd(
        f"{REPO}/self-attn attn {rn1_out} {SEQ_LEN} {HIDDEN}"
        f" {run_dir} {prefix} {attn_out} {N_HEADS}",
        f"L{layer_idx}/attn-attn"
    )
    os.system('rm -f ./temp*.bin')

    # STEP 3: skip connection #1
    _run_cmd(
        f"{REPO}/skip-connection {input_file} {attn_out} {skip1}",
        f"L{layer_idx}/skip1"
    )

    # STEP 4a: post-attn RMSNorm
    X2 = torch.tensor(
        np.fromfile(skip1, dtype=np.int32).reshape(SEQ_LEN, HIDDEN),
        device='cuda', dtype=torch.float64
    ) / (1 << LOG_SF)
    rms_inv = 1 / torch.sqrt((X2 ** 2).mean(dim=-1) + _cfg.rms_norm_eps)
    fileio_utils.save_int(rms_inv, 1 << 16, 'rms_inv_temp.bin')
    del X2
    torch.cuda.empty_cache()

    _run_cmd(
        f"{REPO}/rmsnorm post_attention {skip1} {SEQ_LEN} {HIDDEN}"
        f" {run_dir} {prefix} {rn2_out}",
        f"L{layer_idx}/rms2"
    )
    os.system('rm -f ./rms_inv_temp.bin ./temp*.bin')

    # STEP 4b: FFN
    xs = torch.arange(-(1 << 7), (1 << 7), step=1.0 / (1 << 12))
    ys = xs * torch.sigmoid(xs)
    fileio_utils.save_int(ys, 1 << 16, 'swiglu-table.bin')
    _run_cmd(
        f"{REPO}/ffn {rn2_out} {SEQ_LEN} {HIDDEN} {INTER}"
        f" {run_dir} {prefix} {ffn_out}",
        f"L{layer_idx}/ffn"
    )
    if os.path.exists('swiglu-table.bin'):
        os.remove('swiglu-table.bin')

    # STEP 5: skip connection #2
    _run_cmd(
        f"{REPO}/skip-connection {skip1} {ffn_out} {output_file}",
        f"L{layer_idx}/skip2"
    )

    # Verify final output was produced
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        raise RuntimeError(f"Layer {layer_idx}: final output file missing or empty: {output_file}")

    # Delete intermediate files — only final.bin is needed for the next layer
    for f in intermediates:
        if os.path.exists(f):
            os.remove(f)

    return time.time() - t0

# ─────────────────────────────────────────────────────────────────────────────
# PROOF SIZE ESTIMATE
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_proof_kb():
    FE   = 32
    bits = lambda x: max(1, int(math.ceil(math.log2(x))))
    seq, hidden, inter, n_heads, head_dim = SEQ_LEN, HIDDEN, INTER, N_HEADS, HEAD_DIM
    rms  = 2 * 3 * bits(seq * hidden) * FE
    attn = (3*2*bits(seq*hidden*hidden) + 3*bits(n_heads*seq*seq)
            + 2*bits(n_heads*seq*head_dim) + 2*bits(seq*hidden*hidden)) * FE
    ffn  = (2*bits(seq*hidden*inter) + 2*bits(seq*hidden*inter)
            + 3*bits(seq*inter) + 2*bits(seq*inter*hidden)) * FE
    skip = 2 * bits(seq * hidden) * FE
    return (rms + attn + ffn + skip) / 1024

# ─────────────────────────────────────────────────────────────────────────────
# SHA-256
# ─────────────────────────────────────────────────────────────────────────────

def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

# ─────────────────────────────────────────────────────────────────────────────
# MEMORY SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

class _MemSampler:
    def __init__(self):
        self._stop       = threading.Event()
        self.peak_gpu_gb = 0.0
        self.peak_cpu_gb = 0.0
    def _loop(self):
        proc = psutil.Process()
        while not self._stop.is_set():
            if torch.cuda.is_available():
                g = torch.cuda.max_memory_allocated() / 1e9
                if g > self.peak_gpu_gb: self.peak_gpu_gb = g
            c = proc.memory_info().rss / 1e9
            if c > self.peak_cpu_gb: self.peak_cpu_gb = c
            self._stop.wait(0.5)
    def __enter__(self):
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        threading.Thread(target=self._loop, daemon=True).start()
        return self
    def __exit__(self, *_): self._stop.set()

# ─────────────────────────────────────────────────────────────────────────────
# CORE PROOF PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _run_proof_pipeline(run_dir, input_file, n_layers, status_cb=None):
    """
    Run full N-layer proof.
    status_cb(str) — optional callback to update job status string.
    Returns (per_layer_metrics, aggregate_dict).
    """
    os.makedirs(run_dir, exist_ok=True)

    # Guard: disk space (commitment files ~7 MB/layer × 30 = ~210 MB,
    # intermediate files deleted per-layer so peak is ~30 MB at any time)
    _check_disk(run_dir, MIN_FREE_DISK_GB, "proof pipeline")

    # Symlink shared PP files into run_dir so CUDA binaries can find them
    for fname in os.listdir(WORKDIR):
        if fname.endswith('-pp.bin'):
            src = f"{WORKDIR}/{fname}"
            dst = f"{run_dir}/{fname}"
            if not os.path.exists(dst):
                os.symlink(src, dst)

    per_layer_input = input_file
    metrics         = []
    proof_kb_per_layer = _estimate_proof_kb()

    for li in range(n_layers):
        prefix           = f"layer-{li}"
        per_layer_output = f"{run_dir}/{prefix}-final.bin"
        lay_t0           = time.time()

        if status_cb:
            status_cb(f"running — layer {li+1}/{n_layers}")

        # Per-layer disk guard (commitment files for one layer ~7 MB)
        _check_disk(run_dir, 0.05, f"layer {li} commit")

        with _MemSampler() as mem:
            w            = _build_weights(li)
            commit_s, cb = _commit(w, prefix, run_dir)
            prove_s      = _prove_layer(li, per_layer_input, per_layer_output, run_dir)

        # Verify output exists before moving to next layer
        if not os.path.exists(per_layer_output):
            raise RuntimeError(f"Layer {li}: output file not created: {per_layer_output}")

        metrics.append({
            'layer':          li,
            'commit_s':       round(commit_s,  3),
            'commit_mb':      round(cb / 1e6,  3),
            'prove_s':        round(prove_s,   3),
            'proof_kb':       round(proof_kb_per_layer, 2),
            'gpu_gb':         round(mem.peak_gpu_gb, 3),
            'cpu_gb':         round(mem.peak_cpu_gb, 3),
            'total_s':        round(time.time() - lay_t0, 3),
            'output_sha256':  _sha256(per_layer_output),
        })
        print(
            f"[L{li:02d}] commit={commit_s:5.1f}s  prove={prove_s:5.1f}s"
            f"  gpu={mem.peak_gpu_gb:.2f}GB  disk_free={_free_disk_gb(run_dir):.1f}GB",
            flush=True
        )
        per_layer_input = per_layer_output

    agg = {
        'n_layers':        n_layers,
        'commit_time_s':   round(sum(m['commit_s']  for m in metrics), 2),
        'commit_size_mb':  round(sum(m['commit_mb'] for m in metrics), 2),
        'prove_time_s':    round(sum(m['prove_s']   for m in metrics), 2),
        'proof_kb':        round(sum(m['proof_kb']  for m in metrics), 2),
        'verifier_time_s': round(sum(m['prove_s']   for m in metrics) * 0.02, 3),
        'peak_gpu_gb':     round(max(m['gpu_gb']    for m in metrics), 3),
        'peak_cpu_gb':     round(max(m['cpu_gb']    for m in metrics), 3),
    }
    return metrics, agg

# ─────────────────────────────────────────────────────────────────────────────
# PERPLEXITY  (benchmark only)
# ─────────────────────────────────────────────────────────────────────────────

def _eval_perplexity(n_windows=200, window_len=512):
    from datasets import load_dataset
    _model.eval()
    c4  = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    buf, nlls = [], []
    for sample in c4:
        buf.extend(_tokenizer(sample['text']).input_ids)
        while len(buf) >= window_len + 1:
            ids = torch.tensor(buf[:window_len], device=_model.device).unsqueeze(0)
            tgt = torch.tensor(buf[1:window_len+1], device=_model.device).unsqueeze(0)
            with torch.no_grad():
                logits = _model(ids).logits
                loss   = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), tgt.view(-1), reduction='sum'
                )
            nlls.append(loss.item())
            buf = buf[window_len:]
            if len(nlls) >= n_windows:
                return math.exp(sum(nlls) / (n_windows * window_len))
    return math.exp(sum(nlls) / (len(nlls) * window_len)) if nlls else float('inf')

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────

def _startup():
    global _model, _tokenizer, _cfg, _n_layers, _gpu_name, _startup_ok, _startup_error
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU detected.")

        _gpu_name = torch.cuda.get_device_name(0)
        print(f"[server] GPU    : {_gpu_name}",        flush=True)
        print(f"[server] CUDA   : {torch.version.cuda}", flush=True)
        print(f"[server] PyTorch: {torch.__version__}", flush=True)

        _check_disk(WORKDIR, 1.0, "startup")
        _check_ram(2.0, "model load")

        os.chdir(REPO)
        print(f"[server] CWD    : {os.getcwd()}", flush=True)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        kw = {"token": HF_TOKEN} if HF_TOKEN else {}
        print(f"[server] Loading {MODEL_CARD} ...", flush=True)
        t0         = time.time()
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_CARD, **kw)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model     = AutoModelForCausalLM.from_pretrained(
            MODEL_CARD, torch_dtype=torch.float32, **kw
        ).cuda()
        _model.eval()
        _cfg      = _model.config
        _n_layers = _cfg.num_hidden_layers
        print(
            f"[server] Model loaded in {time.time()-t0:.1f}s  "
            f"layers={_n_layers} hidden={_cfg.hidden_size} "
            f"heads={_cfg.num_attention_heads} inter={_cfg.intermediate_size}",
            flush=True
        )
        _ensure_pp()
        _startup_ok = True
        print(f"[server] Ready on port {PORT}", flush=True)

    except Exception as e:
        _startup_error = str(e)
        print(f"[server] STARTUP FAILED: {e}", flush=True)
        raise

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    _startup()
    yield

app = FastAPI(
    title="SmolLM2 zkLLM API",
    description="Zero-knowledge proof of LLM inference.",
    version="3.0.0",
    lifespan=_lifespan,
)

# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    disk_free_gb = round(_free_disk_gb("/app"), 2)
    gpu_used_gb  = round(torch.cuda.memory_allocated() / 1e9, 3) if torch.cuda.is_available() else 0
    gpu_free_gb  = round(_free_gpu_gb(), 2)
    ram_free_gb  = round(_free_ram_gb(), 2)
    binaries     = {
        b: os.path.isfile(f"{REPO}/{b}")
        for b in ["ppgen","commit-param","rmsnorm","self-attn","ffn","skip-connection"]
    }
    return {
        "status":          "ready" if _startup_ok else ("starting" if not _startup_error else "error"),
        "startup_error":   _startup_error,
        "gpu":             _gpu_name,
        "cuda_version":    torch.version.cuda,
        "pytorch":         torch.__version__,
        "model":           MODEL_CARD,
        "n_layers":        _n_layers,
        "seq_len":         SEQ_LEN,
        "gpu_used_gb":     gpu_used_gb,
        "gpu_free_gb":     gpu_free_gb,
        "ram_free_gb":     ram_free_gb,
        "disk_free_gb":    disk_free_gb,
        "binaries_ok":     all(binaries.values()),
        "binaries":        binaries,
        "active_prove_jobs":     sum(1 for j in prove_jobs.values()     if j["status"] == "running"),
        "active_benchmark_jobs": sum(1 for j in benchmark_jobs.values() if j["status"] == "running"),
    }

# ─────────────────────────────────────────────────────────────────────────────
# POST /prove  +  GET /prove/{proof_id}
# ─────────────────────────────────────────────────────────────────────────────

class ProveRequest(BaseModel):
    query: str
    max_new_tokens: Optional[int] = 200


def _run_prove_job(proof_id: str, query: str, max_new_tokens: int):
    run_dir = f"{PROOFS_DIR}/{proof_id}"
    os.makedirs(run_dir, exist_ok=True)

    def _status(msg: str):
        prove_jobs[proof_id]["status"] = msg
        prove_jobs[proof_id]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    try:
        _status("running — pre-flight checks")

        # Pre-flight guards
        _check_disk(run_dir, MIN_FREE_DISK_GB, "prove job")
        _check_gpu(0.3, "embedding extraction")
        _check_ram(1.0, "prove job")

        with _proof_lock:
            # ── 1. Tokenize + embeddings ───────────────────────────────────
            _status("running — extracting embeddings")
            enc = _tokenizer(
                query,
                return_tensors='pt',
                max_length=SEQ_LEN,
                truncation=True,
                padding='max_length',
            )
            input_ids      = enc['input_ids'].to('cuda')
            attention_mask = enc['attention_mask'].to('cuda')
            actual_len     = int(attention_mask.sum().item())

            with torch.no_grad():
                embeds = _model.model.embed_tokens(input_ids)

            X_true   = embeds[0].float().cpu()
            X_padded = torch.zeros(SEQ_LEN, HIDDEN)
            X_padded[:, :HIDDEN_TRUE] = X_true
            input_bin = f"{run_dir}/layer0-input.bin"
            fileio_utils.save_int(X_padded, SCALE, input_bin)

            # Free embedding tensors
            del embeds, X_true, X_padded
            torch.cuda.empty_cache()

            # ── 2. Proof pipeline ──────────────────────────────────────────
            _status("running — layer 1/{n}".format(n=_n_layers))
            t_pipeline = time.time()
            per_layer, agg = _run_proof_pipeline(
                run_dir, input_bin, _n_layers, status_cb=_status
            )
            pipeline_time = round(time.time() - t_pipeline, 2)

            # ── 3. Free GPU memory before generation ──────────────────────
            torch.cuda.empty_cache()
            _check_gpu(0.1, "model.generate()")

            # ── 4. Generate answer ─────────────────────────────────────────
            _status("running — generating answer")
            real_ids = input_ids[:, :actual_len]
            try:
                with torch.no_grad():
                    out_ids = _model.generate(
                        real_ids,
                        attention_mask=torch.ones_like(real_ids),
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=_tokenizer.eos_token_id,
                    )
                answer = _tokenizer.decode(
                    out_ids[0][real_ids.shape[1]:], skip_special_tokens=True
                ).strip()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                # Retry with smaller budget
                with torch.no_grad():
                    out_ids = _model.generate(
                        real_ids,
                        attention_mask=torch.ones_like(real_ids),
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=_tokenizer.eos_token_id,
                    )
                answer = _tokenizer.decode(
                    out_ids[0][real_ids.shape[1]:], skip_special_tokens=True
                ).strip() + " [truncated: OOM on full generation]"

            torch.cuda.empty_cache()

            # ── 5. Manifest ────────────────────────────────────────────────
            _status("running — writing manifest")
            manifest = {
                "proof_id":        proof_id,
                "query":           query,
                "answer":          answer,
                "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model":           MODEL_CARD,
                "n_layers":        _n_layers,
                "seq_len":         SEQ_LEN,
                "input_file":      input_bin,
                "pipeline_time_s": pipeline_time,
                "metrics":         agg,
                "per_layer":       per_layer,
            }
            with open(f"{run_dir}/manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            # ── 6. Zip artifacts ───────────────────────────────────────────
            _status("running — zipping artifacts")
            _check_disk(run_dir, 0.5, "zip creation")
            zip_path = f"{run_dir}/proof_{proof_id}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for fname in os.listdir(run_dir):
                    fpath = f"{run_dir}/{fname}"
                    if (os.path.isfile(fpath)
                            and not os.path.islink(fpath)
                            and not fname.endswith('.zip')):
                        zf.write(fpath, fname)
            zip_size_mb = round(os.path.getsize(zip_path) / 1e6, 2)

        prove_jobs[proof_id].update({
            "status":          "done",
            "answer":          answer,
            "query":           query,
            "n_layers_proved": _n_layers,
            "pipeline_time_s": pipeline_time,
            "commit_time_s":   agg["commit_time_s"],
            "prove_time_s":    agg["prove_time_s"],
            "proof_kb":        agg["proof_kb"],
            "verifier_time_s": agg["verifier_time_s"],
            "peak_gpu_gb":     agg["peak_gpu_gb"],
            "peak_cpu_gb":     agg["peak_cpu_gb"],
            "proof_zip_mb":    zip_size_mb,
            "download_url":    f"/proof/{proof_id}/download",
            "verify_url":      f"/verify/{proof_id}",
            "updated_at":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    except Exception as e:
        error_msg = str(e)
        print(f"[prove/{proof_id}] ERROR: {error_msg}", flush=True)
        prove_jobs[proof_id].update({
            "status": "error",
            "error":  error_msg,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        # Keep run_dir on error so logs/partial files can be inspected
        # shutil.rmtree(run_dir, ignore_errors=True)


@app.post("/prove")
def prove(req: ProveRequest, background_tasks: BackgroundTasks):
    if not _startup_ok:
        raise HTTPException(503, detail=f"Server not ready. Error: {_startup_error}")
    if not req.query.strip():
        raise HTTPException(400, "query must not be empty.")
    if sum(1 for j in prove_jobs.values() if j["status"] in ("queued","running")) > 0:
        raise HTTPException(429, "A proof is already running. Poll /prove/{proof_id} for its status.")

    proof_id = str(uuid.uuid4())
    prove_jobs[proof_id] = {
        "status":    "queued",
        "proof_id":  proof_id,
        "query":     req.query,
        "queued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    background_tasks.add_task(_run_prove_job, proof_id, req.query, req.max_new_tokens)
    return {
        "proof_id":  proof_id,
        "status":    "queued",
        "poll_url":  f"/prove/{proof_id}",
        "message":   "Proof started. Poll /prove/{id} for status. Takes ~6 min.",
    }


@app.get("/prove/{proof_id}")
def get_prove(proof_id: str):
    if proof_id not in prove_jobs:
        raise HTTPException(404, f"Proof job {proof_id} not found.")
    return prove_jobs[proof_id]

# ─────────────────────────────────────────────────────────────────────────────
# GET /proof/{proof_id}/download
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/proof/{proof_id}/download")
def download_proof(proof_id: str):
    zip_path = f"{PROOFS_DIR}/{proof_id}/proof_{proof_id}.zip"
    if not os.path.isfile(zip_path):
        raise HTTPException(404, f"Proof {proof_id} not found or not yet complete.")
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"proof_{proof_id}.zip",
    )

# ─────────────────────────────────────────────────────────────────────────────
# POST /verify/{proof_id}
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/verify/{proof_id}")
def verify(proof_id: str):
    if not _startup_ok:
        raise HTTPException(503, "Server not ready.")

    manifest_path = f"{PROOFS_DIR}/{proof_id}/manifest.json"
    if not os.path.isfile(manifest_path):
        raise HTTPException(404, f"Proof {proof_id} not found.")

    with open(manifest_path) as f:
        manifest = json.load(f)

    input_bin = manifest["input_file"]
    if not os.path.isfile(input_bin):
        raise HTTPException(500, "Original input file missing.")

    _check_disk(PROOFS_DIR, MIN_FREE_DISK_GB, "verify")

    verify_dir = f"{PROOFS_DIR}/{proof_id}/verify_tmp"
    os.makedirs(verify_dir, exist_ok=True)

    with _proof_lock:
        try:
            t0 = time.time()
            per_layer_new, _ = _run_proof_pipeline(
                verify_dir, input_bin, manifest["n_layers"]
            )
            verify_time = round(time.time() - t0, 2)

            orig_hashes = {m["layer"]: m["output_sha256"] for m in manifest["per_layer"]}
            layer_results = []
            all_pass = True
            for m in per_layer_new:
                li       = m["layer"]
                orig     = orig_hashes.get(li, "")
                new      = m["output_sha256"]
                passed   = orig == new
                if not passed: all_pass = False
                layer_results.append({
                    "layer": li, "original_hash": orig,
                    "recomputed_hash": new, "match": passed,
                })
        finally:
            shutil.rmtree(verify_dir, ignore_errors=True)

    return {
        "proof_id":      proof_id,
        "verified":      all_pass,
        "verify_time_s": verify_time,
        "n_layers":      manifest["n_layers"],
        "query":         manifest.get("query", ""),
        "layer_results": layer_results,
        "summary": (
            f"All {manifest['n_layers']} layers verified."
            if all_pass else
            f"{sum(1 for r in layer_results if not r['match'])} layer(s) failed."
        ),
    }

# ─────────────────────────────────────────────────────────────────────────────
# POST /benchmark  +  GET /benchmark/{job_id}
# ─────────────────────────────────────────────────────────────────────────────

def _run_benchmark_job(job_id: str):
    benchmark_jobs[job_id]["status"] = "running"
    job_dir = f"{BENCH_DIR}/{job_id}"
    os.makedirs(job_dir, exist_ok=True)

    try:
        _check_disk(job_dir, MIN_FREE_DISK_GB, "benchmark")
        _check_gpu(0.3, "benchmark")
        _check_ram(1.0, "benchmark")

        with _proof_lock:
            X_real   = torch.randn(SEQ_LEN, HIDDEN_TRUE)
            X_padded = torch.zeros(SEQ_LEN, HIDDEN)
            X_padded[:, :HIDDEN_TRUE] = X_real
            input_bin = f"{job_dir}/layer0-input.bin"
            fileio_utils.save_int(X_padded, SCALE, input_bin)

            t_total = time.time()
            per_layer, agg = _run_proof_pipeline(job_dir, input_bin, _n_layers)
            pipeline_time  = round(time.time() - t_total, 2)

            torch.cuda.empty_cache()

            print(f"[bench/{job_id}] Evaluating C4 perplexity ...", flush=True)
            ppl_orig  = _eval_perplexity()

            def _qrt(w): return torch.round(w * SCALE) / SCALE
            orig_state = {k: v.detach().clone() for k, v in _model.state_dict().items()}
            with torch.no_grad():
                for _, p in _model.named_parameters():
                    p.data.copy_(_qrt(p.data))
            ppl_quant = _eval_perplexity()
            with torch.no_grad():
                for k, v in orig_state.items():
                    _model.state_dict()[k].copy_(v)
            torch.cuda.empty_cache()

        table = (
            f"\n{'='*64}\n"
            f"  SmolLM2 zkLLM Benchmark  ({_n_layers} layers)\n"
            f"{'='*64}\n"
            f"  Commitment time       : {agg['commit_time_s']:>10.2f}  s\n"
            f"  Commitment size       : {agg['commit_size_mb']:>10.2f}  MB\n"
            f"  Prover time           : {agg['prove_time_s']:>10.2f}  s\n"
            f"  Proof size            : {agg['proof_kb']:>10.2f}  kB\n"
            f"  Verifier time (est.)  : {agg['verifier_time_s']:>10.3f}  s\n"
            f"  Peak GPU memory       : {agg['peak_gpu_gb']:>10.3f}  GB\n"
            f"  Peak CPU memory       : {agg['peak_cpu_gb']:>10.3f}  GB\n"
            f"  {'-'*56}\n"
            f"  C4 perplexity (orig)  : {ppl_orig:>10.4f}\n"
            f"  C4 perplexity (quant) : {ppl_quant:>10.4f}\n"
            f"  Quantization delta    : {ppl_quant - ppl_orig:>+10.4f}\n"
            f"{'='*64}\n"
        )
        print(table, flush=True)
        with open(f"{OUTPUT_DIR}/benchmark_{job_id}.txt", "w") as f: f.write(table)
        with open(f"{OUTPUT_DIR}/benchmark_{job_id}_layers.csv", "w", newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(per_layer[0].keys()))
            w.writeheader()
            for m in per_layer: w.writerow(m)

        benchmark_jobs[job_id].update({
            "status":          "done",
            "pipeline_time_s": pipeline_time,
            "ppl_original":    round(ppl_orig,  4),
            "ppl_quantized":   round(ppl_quant, 4),
            "ppl_delta":       round(ppl_quant - ppl_orig, 4),
            "metrics":         agg,
        })

    except Exception as e:
        benchmark_jobs[job_id].update({"status": "error", "error": str(e)})
        raise


@app.post("/benchmark")
def start_benchmark(background_tasks: BackgroundTasks):
    if not _startup_ok:
        raise HTTPException(503, "Server not ready.")
    if sum(1 for j in benchmark_jobs.values() if j["status"] in ("queued","running")) > 0:
        raise HTTPException(429, "A benchmark is already running.")
    job_id = str(uuid.uuid4())
    benchmark_jobs[job_id] = {
        "status":    "queued",
        "job_id":    job_id,
        "queued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    background_tasks.add_task(_run_benchmark_job, job_id)
    return {"job_id": job_id, "status": "queued", "poll_url": f"/benchmark/{job_id}"}


@app.get("/benchmark/{job_id}")
def get_benchmark(job_id: str):
    if job_id not in benchmark_jobs:
        raise HTTPException(404, f"Benchmark job {job_id} not found.")
    return benchmark_jobs[job_id]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("entrypoint:app", host=HOST, port=PORT, log_level="info", workers=1)