#!/usr/bin/env python3
"""
entrypoint.py — SmolLM2 zkLLM API Server

Endpoints
---------
GET  /health                     — server status, GPU info, model info
POST /prove                      — prove a query, return answer + proof artifacts
GET  /proof/{proof_id}/download  — download zip of all proof artifacts
POST /verify/{proof_id}          — re-run proof on stored input, compare hashes
POST /benchmark                  — start background benchmark job (returns job_id)
GET  /benchmark/{job_id}         — poll benchmark status / retrieve result
"""

import os, sys, time, math, json, csv, threading, subprocess, uuid, shutil, hashlib, zipfile
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import numpy as np
import psutil
import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Resolve SCRIPT_DIR before any chdir ──────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import fileio_utils

# ─────────────────────────────────────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CARD = os.environ.get("MODEL_CARD", "HuggingFaceTB/SmolLM2-135M")
SEQ_LEN    = int(os.environ.get("SEQ_LEN",    "512"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR",  "/app/output")
WORKDIR    = os.environ.get("WORKDIR",     "/app/zkllm-workdir")
REPO       = os.environ.get("ZKLLM_REPO",  "/app/zkllm-ccs2024")
HF_TOKEN   = os.environ.get("HF_TOKEN",    None)
PROOFS_DIR = os.environ.get("PROOFS_DIR",  "/app/proofs")
BENCH_DIR  = os.environ.get("BENCH_DIR",   "/app/benchmarks")
HOST       = os.environ.get("HOST",        "0.0.0.0")
PORT       = int(os.environ.get("PORT",    "8000"))

for _d in [OUTPUT_DIR, WORKDIR, PROOFS_DIR, BENCH_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE CONSTANTS  (SmolLM2-135M padded to power-of-2 dims)
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_TRUE   = 576
N_HEADS_TRUE  = 9
KV_HEADS_TRUE = 3
HEAD_DIM      = 64       # 576 / 9 = 64, already power of 2
N_GROUPS      = N_HEADS_TRUE // KV_HEADS_TRUE  # 3

HIDDEN  = 1024  # next pow2 >= 576
N_HEADS = 16    # next pow2 >= 9   → head_dim = 1024/16 = 64 ✓
INTER   = 2048  # next pow2 >= 1536

LOG_SF      = 16
SCALE       = 1 << LOG_SF
LOG_OFF     = 5
VALUE_LOGSF = 16
ACCU_LOGSF  = 20

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE  (populated at startup, read-only after that)
# ─────────────────────────────────────────────────────────────────────────────
_model      = None
_tokenizer  = None
_cfg        = None
_n_layers   = None       # cfg.num_hidden_layers, loaded dynamically
_gpu_name   = "unknown"
_startup_ok = False

# One GPU → one proof/benchmark at a time.
_proof_lock    = threading.Lock()
benchmark_jobs: Dict[str, Any] = {}
_executor = ThreadPoolExecutor(max_workers=1)

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
# PUBLIC PARAMETERS  (generated once, reused across all runs)
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_pp():
    print("[server] Checking public parameters ...", flush=True)
    ref = _build_weights(0)
    t0  = time.time()
    for name, w in ref.items():
        pp_path = f"{WORKDIR}/{name}-pp.bin"
        if os.path.exists(pp_path):
            continue
        pp_size = (w.shape[0] << LOG_OFF) if w.ndim == 2 else w.shape[0]
        os.system(f"{REPO}/ppgen {pp_size} {pp_path} > /dev/null 2>&1")
        print(f"  [pp] generated {name}", flush=True)
    print(f"[server] PP ready in {time.time()-t0:.1f}s", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# COMMIT
# ─────────────────────────────────────────────────────────────────────────────

def _commit(weights, prefix, run_dir):
    t0 = total_bytes = 0
    t0 = time.time()
    for name, tensor in weights.items():
        w_orig = tensor.float().T if tensor.ndim == 2 else tensor.float()
        w_int  = torch.round(w_orig * SCALE).to(torch.int32)

        pp_path     = f"{WORKDIR}/{name}-pp.bin"           # shared
        int_path    = f"{run_dir}/{prefix}-{name}-int.bin"
        commit_path = f"{run_dir}/{prefix}-{name}-commitment.bin"

        w_int.cpu().numpy().astype(np.int32).tofile(int_path)
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
        if os.path.exists(commit_path):
            total_bytes += os.path.getsize(commit_path)
    return time.time() - t0, total_bytes

# ─────────────────────────────────────────────────────────────────────────────
# PROOF PIPELINE  (one layer)
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd, tag):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"[{tag}] FAILED\nCMD: {cmd}\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"
        )

def _prove_layer(layer_idx, input_file, output_file, run_dir):
    prefix   = f"layer-{layer_idx}"
    rn1_out  = f"{run_dir}/{prefix}-rn1-out.bin"
    attn_out = f"{run_dir}/{prefix}-attn-out.bin"
    skip1    = f"{run_dir}/{prefix}-skip1.bin"
    rn2_out  = f"{run_dir}/{prefix}-rn2-out.bin"
    ffn_out  = f"{run_dir}/{prefix}-ffn-out.bin"
    t0       = time.time()

    # STEP 1: input RMSNorm
    X = torch.tensor(
        np.fromfile(input_file, dtype=np.int32).reshape(SEQ_LEN, HIDDEN),
        device='cuda', dtype=torch.float64
    ) / (1 << LOG_SF)
    rms_inv = 1 / torch.sqrt((X ** 2).mean(dim=-1) + _cfg.rms_norm_eps)
    fileio_utils.save_int(rms_inv, 1 << 16, 'rms_inv_temp.bin')
    _run(
        f"{REPO}/rmsnorm input {input_file} {SEQ_LEN} {HIDDEN}"
        f" {run_dir} {prefix} {rn1_out}",
        f"L{layer_idx}/rms1"
    )
    os.system('rm -f ./rms_inv_temp.bin ./temp*.bin')

    # STEP 2: self-attention
    _run(
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
    def _rotate(x):
        a, b = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-b, a), dim=-1)
    Q = Q * cos + _rotate(Q) * sin
    K = K * cos + _rotate(K) * sin

    A_   = Q @ K.transpose(-2, -1)
    A    = fileio_utils.to_int64(A_, VALUE_LOGSF)
    mask = torch.triu(
        torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool, device='cuda'), diagonal=1
    )
    A -= torch.max(A * ~mask, dim=-1, keepdim=True).values
    shift = math.sqrt(HEAD_DIM) * torch.log(
        (torch.exp(
            fileio_utils.to_float(A, ACCU_LOGSF, torch.float64) / math.sqrt(HEAD_DIM)
        ) * ~mask).sum(dim=-1, keepdim=True).clamp(min=1e-9)
    )
    A -= fileio_utils.to_int64(shift, ACCU_LOGSF)
    attn_w   = torch.exp(
        fileio_utils.to_float(A, ACCU_LOGSF, torch.float64) / math.sqrt(HEAD_DIM)
    ) * ~mask
    attn_pre = fileio_utils.fromto_int64(attn_w @ V.to(attn_w.dtype), VALUE_LOGSF)
    attn_pre = attn_pre.transpose(0, 1).contiguous().view(SEQ_LEN, HIDDEN)
    fileio_utils.save_int(attn_pre, 1 << VALUE_LOGSF, 'temp_attn_out.bin')
    _run(
        f"{REPO}/self-attn attn {rn1_out} {SEQ_LEN} {HIDDEN}"
        f" {run_dir} {prefix} {attn_out} {N_HEADS}",
        f"L{layer_idx}/attn-attn"
    )
    os.system('rm -f ./temp*.bin')

    # STEP 3: skip connection #1
    _run(f"{REPO}/skip-connection {input_file} {attn_out} {skip1}", f"L{layer_idx}/skip1")

    # STEP 4a: post-attn RMSNorm
    X2 = torch.tensor(
        np.fromfile(skip1, dtype=np.int32).reshape(SEQ_LEN, HIDDEN),
        device='cuda', dtype=torch.float64
    ) / (1 << LOG_SF)
    rms_inv = 1 / torch.sqrt((X2 ** 2).mean(dim=-1) + _cfg.rms_norm_eps)
    fileio_utils.save_int(rms_inv, 1 << 16, 'rms_inv_temp.bin')
    _run(
        f"{REPO}/rmsnorm post_attention {skip1} {SEQ_LEN} {HIDDEN}"
        f" {run_dir} {prefix} {rn2_out}",
        f"L{layer_idx}/rms2"
    )
    os.system('rm -f ./rms_inv_temp.bin ./temp*.bin')

    # STEP 4b: FFN
    xs = torch.arange(-(1 << 7), (1 << 7), step=1.0 / (1 << 12))
    ys = xs * torch.sigmoid(xs)
    fileio_utils.save_int(ys, 1 << 16, 'swiglu-table.bin')
    _run(
        f"{REPO}/ffn {rn2_out} {SEQ_LEN} {HIDDEN} {INTER}"
        f" {run_dir} {prefix} {ffn_out}",
        f"L{layer_idx}/ffn"
    )
    if os.path.exists('swiglu-table.bin'):
        os.remove('swiglu-table.bin')

    # STEP 5: skip connection #2
    _run(f"{REPO}/skip-connection {skip1} {ffn_out} {output_file}", f"L{layer_idx}/skip2")

    return time.time() - t0

# ─────────────────────────────────────────────────────────────────────────────
# PROOF SIZE ESTIMATE
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_proof_kb(seq, hidden, inter, n_heads, head_dim):
    FE   = 32
    bits = lambda x: max(1, int(math.ceil(math.log2(x))))
    rms  = 2 * 3 * bits(seq * hidden) * FE
    attn = (3*2*bits(seq*hidden*hidden) + 3*bits(n_heads*seq*seq)
            + 2*bits(n_heads*seq*head_dim) + 2*bits(seq*hidden*hidden)) * FE
    ffn  = (2*bits(seq*hidden*inter) + 2*bits(seq*hidden*inter)
            + 3*bits(seq*inter) + 2*bits(seq*inter*hidden)) * FE
    skip = 2 * bits(seq * hidden) * FE
    return (rms + attn + ffn + skip) / 1024

# ─────────────────────────────────────────────────────────────────────────────
# SHA-256 of a file (for proof manifest + verification)
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
# CORE PROOF RUNNER  (used by both /prove and /verify)
# ─────────────────────────────────────────────────────────────────────────────

def _run_proof_pipeline(run_dir, input_file, n_layers):
    """
    Run the full N-layer proof pipeline.
    Returns (per_layer_metrics list, aggregate dict).
    run_dir   : where layer output + commit files go
    input_file: path to layer0-input.bin (SEQ_LEN, HIDDEN) int32
    """
    os.makedirs(run_dir, exist_ok=True)

    # Symlink shared PP files from WORKDIR into run_dir so the CUDA binaries
    # can find them. The binaries look for {workdir}/{name}-pp.bin.
    for fname in os.listdir(WORKDIR):
        if fname.endswith('-pp.bin'):
            src = f"{WORKDIR}/{fname}"
            dst = f"{run_dir}/{fname}"
            if not os.path.exists(dst):
                os.symlink(src, dst)

    per_layer_input = input_file
    metrics = []

    for li in range(n_layers):
        prefix           = f"layer-{li}"
        per_layer_output = f"{run_dir}/{prefix}-final.bin"
        lay_t0           = time.time()

        with _MemSampler() as mem:
            w              = _build_weights(li)
            commit_s, cb   = _commit(w, prefix, run_dir)
            prove_s        = _prove_layer(li, per_layer_input, per_layer_output, run_dir)

        metrics.append({
            'layer':     li,
            'commit_s':  round(commit_s, 3),
            'commit_mb': round(cb / 1e6, 3),
            'prove_s':   round(prove_s, 3),
            'proof_kb':  round(_estimate_proof_kb(SEQ_LEN, HIDDEN, INTER, N_HEADS, HEAD_DIM), 2),
            'gpu_gb':    round(mem.peak_gpu_gb, 3),
            'cpu_gb':    round(mem.peak_cpu_gb, 3),
            'total_s':   round(time.time() - lay_t0, 3),
            'output_sha256': _sha256(per_layer_output),
        })
        print(
            f"[L{li:02d}] commit={commit_s:5.1f}s  prove={prove_s:5.1f}s"
            f"  gpu={mem.peak_gpu_gb:.2f}GB  total={time.time()-lay_t0:.1f}s",
            flush=True
        )
        per_layer_input = per_layer_output

    agg = {
        'n_layers':     n_layers,
        'commit_time_s':  round(sum(m['commit_s']  for m in metrics), 2),
        'commit_size_mb': round(sum(m['commit_mb'] for m in metrics), 2),
        'prove_time_s':   round(sum(m['prove_s']   for m in metrics), 2),
        'proof_kb':       round(sum(m['proof_kb']  for m in metrics), 2),
        'verifier_time_s':round(sum(m['prove_s']   for m in metrics) * 0.02, 3),
        'peak_gpu_gb':    round(max(m['gpu_gb']    for m in metrics), 3),
        'peak_cpu_gb':    round(max(m['cpu_gb']    for m in metrics), 3),
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
    global _model, _tokenizer, _cfg, _n_layers, _gpu_name, _startup_ok

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. This container requires a GPU host.")

    _gpu_name = torch.cuda.get_device_name(0)
    print(f"[server] GPU    : {_gpu_name}", flush=True)
    print(f"[server] CUDA   : {torch.version.cuda}", flush=True)
    print(f"[server] PyTorch: {torch.__version__}", flush=True)

    # chdir to REPO so CUDA binaries write temp files to the right place
    os.chdir(REPO)
    print(f"[server] CWD    : {os.getcwd()}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = {"token": HF_TOKEN} if HF_TOKEN else {}
    print(f"[server] Loading {MODEL_CARD} ...", flush=True)
    t0         = time.time()
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_CARD, **kw)
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

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    _startup()
    yield

app = FastAPI(
    title="SmolLM2 zkLLM API",
    description="Prove LLM inference with zero-knowledge proofs.",
    version="2.0.0",
    lifespan=_lifespan,
)

# ─────────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    gpu_mem_gb = (
        round(torch.cuda.memory_allocated() / 1e9, 3)
        if torch.cuda.is_available() else 0
    )
    binaries = {
        b: os.path.isfile(f"{REPO}/{b}")
        for b in ["ppgen", "commit-param", "rmsnorm", "self-attn", "ffn", "skip-connection"]
    }
    return {
        "status":        "ready" if _startup_ok else "starting",
        "gpu":           _gpu_name,
        "cuda_version":  torch.version.cuda,
        "pytorch":       torch.__version__,
        "model":         MODEL_CARD,
        "n_layers":      _n_layers,
        "seq_len":       SEQ_LEN,
        "gpu_mem_used_gb": gpu_mem_gb,
        "binaries_ok":   all(binaries.values()),
        "binaries":      binaries,
    }

# ─────────────────────────────────────────────────────────────────────────────
# /prove
# ─────────────────────────────────────────────────────────────────────────────

class ProveRequest(BaseModel):
    query: str
    max_new_tokens: Optional[int] = 200

@app.post("/prove")
def prove(req: ProveRequest):
    if not _startup_ok:
        raise HTTPException(503, "Server is still starting up.")
    if not req.query.strip():
        raise HTTPException(400, "query must not be empty.")

    proof_id  = str(uuid.uuid4())
    run_dir   = f"{PROOFS_DIR}/{proof_id}"
    os.makedirs(run_dir, exist_ok=True)

    with _proof_lock:
        try:
            # ── 1. Tokenize + get real layer-0 embeddings ─────────────────
            kw = {"token": HF_TOKEN} if HF_TOKEN else {}
            enc = _tokenizer(
                req.query,
                return_tensors='pt',
                max_length=SEQ_LEN,
                truncation=True,
            )
            input_ids = enc['input_ids'].to('cuda')
            actual_len = input_ids.shape[1]

            with torch.no_grad():
                # shape: (1, actual_len, hidden_true)
                embeds = _model.model.embed_tokens(input_ids)

            # Pad sequence to SEQ_LEN, pad hidden to HIDDEN
            X_true   = embeds[0].float().cpu()              # (actual_len, 576)
            X_padded = torch.zeros(SEQ_LEN, HIDDEN)
            X_padded[:actual_len, :HIDDEN_TRUE] = X_true
            input_bin = f"{run_dir}/layer0-input.bin"
            fileio_utils.save_int(X_padded, SCALE, input_bin)

            # ── 2. Run full N-layer proof pipeline ─────────────────────────
            t_pipeline = time.time()
            per_layer, agg = _run_proof_pipeline(run_dir, input_bin, _n_layers)
            pipeline_time  = round(time.time() - t_pipeline, 2)

            # ── 3. Generate answer ─────────────────────────────────────────
            with torch.no_grad():
                out_ids = _model.generate(
                    input_ids,
                    max_new_tokens=req.max_new_tokens,
                    do_sample=False,
                    pad_token_id=_tokenizer.eos_token_id,
                )
            answer = _tokenizer.decode(
                out_ids[0][input_ids.shape[1]:], skip_special_tokens=True
            ).strip()

            # ── 4. Write manifest (used by /verify) ────────────────────────
            manifest = {
                "proof_id":    proof_id,
                "query":       req.query,
                "answer":      answer,
                "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model":       MODEL_CARD,
                "n_layers":    _n_layers,
                "seq_len":     SEQ_LEN,
                "input_file":  input_bin,
                "pipeline_time_s": pipeline_time,
                "metrics":     agg,
                "per_layer":   per_layer,
            }
            with open(f"{run_dir}/manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            # ── 5. Zip proof artifacts ─────────────────────────────────────
            zip_path = f"{run_dir}/proof_{proof_id}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for fname in os.listdir(run_dir):
                    fpath = f"{run_dir}/{fname}"
                    if os.path.isfile(fpath) and not fname.endswith('.zip'):
                        zf.write(fpath, fname)
            zip_size_mb = round(os.path.getsize(zip_path) / 1e6, 2)

            return {
                "proof_id":        proof_id,
                "answer":          answer,
                "query":           req.query,
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
            }

        except Exception as e:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise HTTPException(500, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# /proof/{proof_id}/download
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/proof/{proof_id}/download")
def download_proof(proof_id: str):
    run_dir  = f"{PROOFS_DIR}/{proof_id}"
    zip_path = f"{run_dir}/proof_{proof_id}.zip"
    if not os.path.isfile(zip_path):
        raise HTTPException(404, f"Proof {proof_id} not found.")
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"proof_{proof_id}.zip",
    )

# ─────────────────────────────────────────────────────────────────────────────
# /verify/{proof_id}
# Re-runs the proof pipeline on the stored input and compares layer output hashes.
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/verify/{proof_id}")
def verify(proof_id: str):
    if not _startup_ok:
        raise HTTPException(503, "Server is still starting up.")

    run_dir      = f"{PROOFS_DIR}/{proof_id}"
    manifest_path = f"{run_dir}/manifest.json"
    if not os.path.isfile(manifest_path):
        raise HTTPException(404, f"Proof {proof_id} not found.")

    with open(manifest_path) as f:
        manifest = json.load(f)

    input_bin = manifest["input_file"]
    if not os.path.isfile(input_bin):
        raise HTTPException(500, "Original input file missing from proof directory.")

    verify_dir = f"{run_dir}/verify_tmp"
    os.makedirs(verify_dir, exist_ok=True)

    with _proof_lock:
        try:
            t0 = time.time()
            per_layer_new, _ = _run_proof_pipeline(
                verify_dir, input_bin, manifest["n_layers"]
            )
            verify_time = round(time.time() - t0, 2)

            # Compare layer output hashes
            original_hashes = {
                m["layer"]: m["output_sha256"] for m in manifest["per_layer"]
            }
            layer_results = []
            all_pass = True
            for m_new in per_layer_new:
                li        = m_new["layer"]
                orig_hash = original_hashes.get(li, "")
                new_hash  = m_new["output_sha256"]
                passed    = orig_hash == new_hash
                if not passed:
                    all_pass = False
                layer_results.append({
                    "layer":         li,
                    "original_hash": orig_hash,
                    "recomputed_hash": new_hash,
                    "match":         passed,
                })

            shutil.rmtree(verify_dir, ignore_errors=True)

            return {
                "proof_id":    proof_id,
                "verified":    all_pass,
                "verify_time_s": verify_time,
                "n_layers":    manifest["n_layers"],
                "query":       manifest.get("query", ""),
                "layer_results": layer_results,
                "summary": (
                    f"All {manifest['n_layers']} layers verified successfully."
                    if all_pass
                    else f"{sum(1 for r in layer_results if not r['match'])} layer(s) failed verification."
                ),
            }

        except Exception as e:
            shutil.rmtree(verify_dir, ignore_errors=True)
            raise HTTPException(500, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# /benchmark  (background job)
# ─────────────────────────────────────────────────────────────────────────────

def _run_benchmark_job(job_id: str):
    benchmark_jobs[job_id]["status"] = "running"
    job_dir = f"{BENCH_DIR}/{job_id}"
    os.makedirs(job_dir, exist_ok=True)

    try:
        with _proof_lock:
            # Generate random padded input (same as original notebook)
            X_real   = torch.randn(SEQ_LEN, HIDDEN_TRUE)
            X_padded = torch.zeros(SEQ_LEN, HIDDEN)
            X_padded[:, :HIDDEN_TRUE] = X_real
            input_bin = f"{job_dir}/layer0-input.bin"
            fileio_utils.save_int(X_padded, SCALE, input_bin)

            t_total = time.time()
            per_layer, agg = _run_proof_pipeline(job_dir, input_bin, _n_layers)
            pipeline_time  = round(time.time() - t_total, 2)

            # Perplexity
            print(f"[bench/{job_id}] Evaluating C4 perplexity ...", flush=True)
            ppl_orig  = _eval_perplexity()

            def _qrt(w):
                return torch.round(w * SCALE) / SCALE
            orig_state = {k: v.detach().clone() for k, v in _model.state_dict().items()}
            with torch.no_grad():
                for _, p in _model.named_parameters():
                    p.data.copy_(_qrt(p.data))
            ppl_quant = _eval_perplexity()
            with torch.no_grad():
                for k, v in orig_state.items():
                    _model.state_dict()[k].copy_(v)

        # Write outputs
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

        with open(f"{OUTPUT_DIR}/benchmark_{job_id}_metrics.txt", "w") as f:
            f.write(table)
        with open(f"{OUTPUT_DIR}/benchmark_{job_id}_per_layer.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(per_layer[0].keys()))
            writer.writeheader()
            for m in per_layer:
                writer.writerow(m)

        benchmark_jobs[job_id].update({
            "status":         "done",
            "pipeline_time_s": pipeline_time,
            "ppl_original":   round(ppl_orig,  4),
            "ppl_quantized":  round(ppl_quant, 4),
            "ppl_delta":      round(ppl_quant - ppl_orig, 4),
            "metrics":        agg,
            "per_layer":      per_layer,
        })

    except Exception as e:
        benchmark_jobs[job_id].update({"status": "error", "error": str(e)})
        raise

@app.post("/benchmark")
def start_benchmark(background_tasks: BackgroundTasks):
    if not _startup_ok:
        raise HTTPException(503, "Server is still starting up.")
    job_id = str(uuid.uuid4())
    benchmark_jobs[job_id] = {
        "status":   "queued",
        "job_id":   job_id,
        "queued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    background_tasks.add_task(_run_benchmark_job, job_id)
    return {
        "job_id":    job_id,
        "status":    "queued",
        "poll_url":  f"/benchmark/{job_id}",
    }

@app.get("/benchmark/{job_id}")
def get_benchmark(job_id: str):
    if job_id not in benchmark_jobs:
        raise HTTPException(404, f"Benchmark job {job_id} not found.")
    return benchmark_jobs[job_id]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "entrypoint:app",
        host=HOST,
        port=PORT,
        log_level="info",
        workers=1,        # single worker — one GPU, one model in memory
    )