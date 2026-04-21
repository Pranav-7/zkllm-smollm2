FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    HF_HOME=/app/hf_cache \
    MODEL_CARD=HuggingFaceTB/SmolLM2-135M \
    SEQ_LEN=512 \
    OUTPUT_DIR=/app/output \
    WORKDIR=/app/zkllm-workdir \
    ZKLLM_REPO=/app/zkllm-ccs2024 \
    PROOFS_DIR=/app/proofs \
    BENCH_DIR=/app/benchmarks \
    HOST=0.0.0.0 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git make python3 python3-pip python3-dev \
        ca-certificates wget curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN rm -f /usr/lib/python3*/EXTERNALLY-MANAGED \
    && pip3 install --upgrade pip \
    && pip3 install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.3.1+cu121 \
    && pip3 install -r /app/requirements.txt

ARG ZKLLM_COMMIT=993311ea8e2346b90efacb35337b2e3cfe0d6f8d
RUN git clone https://github.com/jvhs0706/zkllm-ccs2024.git /app/zkllm-ccs2024 \
    && cd /app/zkllm-ccs2024 \
    && git checkout ${ZKLLM_COMMIT}

COPY patches/self-attn.cu /app/zkllm-ccs2024/self-attn.cu
COPY patches/patch_ffn.sh /app/patches/patch_ffn.sh
RUN bash /app/patches/patch_ffn.sh /app/zkllm-ccs2024/ffn.cu

COPY compile_zkllm.sh /app/compile_zkllm.sh
COPY entrypoint.py    /app/entrypoint.py
COPY fileio_utils.py  /app/fileio_utils.py
RUN chmod +x /app/compile_zkllm.sh

RUN mkdir -p /app/output /app/zkllm-workdir /app/hf_cache /app/proofs /app/benchmarks

EXPOSE 8000

ENTRYPOINT ["/bin/bash", "-c", "/app/compile_zkllm.sh && python3 -u /app/entrypoint.py"]
