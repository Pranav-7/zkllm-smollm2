# SmolLM2-135M zkLLM — From Notebook to Docker Hub to H200

A complete, battle-hardened guide. No Kaniko. No Docker-in-Docker.
Uses GitHub Actions with a real Docker daemon to build and push your image,
then runs it on a RunPod H200 GPU pod.

---

## 1. What this guide does

```
Your laptop (slow net)
       |
       | git push  (only your code, a few KB)
       v
GitHub Actions runner  ←  real Docker daemon, fat pipe to Docker Hub
       |
       | docker build + push  (~20 min, free)
       v
Docker Hub  (public image, ~6-8 GB)
       |
       | docker pull (RunPod pulls it over its own fat pipe, not yours)
       v
RunPod H200  →  compile CUDA binaries  →  run 30-layer zkLLM proof
```

Your laptop only ever pushes a few kilobytes of source code.
GitHub's servers do the heavy lifting.

---

## 2. Folder structure

Create this exact layout. Every file is explained in section 3.

```
zkllm-smollm2/
├── .github/
│   └── workflows/
│       └── build-push.yml      ← GitHub Actions: build + push to Docker Hub
├── patches/
│   ├── self-attn.cu            ← full replacement for upstream self-attn.cu
│   └── patch_ffn.sh            ← regex patch for ffn.cu (shrinks swiglu table)
├── Dockerfile                  ← image build spec
├── requirements.txt            ← Python deps with pinned versions
├── compile_zkllm.sh            ← runs at container start; detects GPU arch + compiles
├── entrypoint.py               ← full notebook logic as a Python script
├── fileio_utils.py             ← fixed-point I/O helpers (imported by entrypoint.py)
├── docker-compose.yml          ← for Phala deployment later
└── .dockerignore               ← keeps build context lean
```

All the files are included in this repository. Do NOT change filenames.

---

## 3. Prerequisites

You need:
- A GitHub account (free). Your code will live in a GitHub repository.
- A Docker Hub account (free). Your image will be pushed here.
- A RunPod account with at least $5 of credit. For testing on H200.

You do NOT need:
- Docker installed locally.
- Any GPU on your laptop.
- Fast internet (you only push source code, never the image).

---

## 4. One-time setup

### 4a. Create a GitHub repository

1. Go to github.com → click the + button → New repository.
2. Name it: `zkllm-smollm2`
3. Set visibility to Private (or Public — your choice).
4. Do NOT initialize with a README (you will push your own files).
5. Click Create repository.

Copy the repository URL shown. It looks like:
`https://github.com/YOUR_USERNAME/zkllm-smollm2.git`

### 4b. Create a Docker Hub access token

You need a token (not your password) for GitHub Actions to push images.

1. Log in at hub.docker.com.
2. Click your avatar in the top-right corner → Account Settings.
3. Click Personal access tokens in the left sidebar.
4. Click Generate new token.
5. Name: `github-actions`
6. Permissions: Read, Write, Delete.
7. Click Generate.
8. Copy the token immediately — you cannot view it again.

The token looks like: `dckr_pat_xxxxxxxxxxxxxxxxxxxxxxxxxx`

### 4c. Add secrets to your GitHub repository

GitHub Actions uses secrets to authenticate with Docker Hub without
hardcoding credentials in your code.

1. Go to your GitHub repository.
2. Click Settings (top menu) → Secrets and variables → Actions.
3. Click New repository secret and add these two secrets:

   Name: `DOCKERHUB_USERNAME`
   Value: your Docker Hub username (e.g. `pranav6773`)

   Name: `DOCKERHUB_TOKEN`
   Value: the access token you just copied (the `dckr_pat_...` string)

That is all the one-time setup required.

---

## 5. Push your code

On your local machine, from inside the `zkllm-smollm2/` folder:

```bash
git init
git add .
git commit -m "initial: SmolLM2-135M zkLLM prover"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/zkllm-smollm2.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

That push triggers the GitHub Actions workflow automatically.

---

## 6. Watch the build

1. Go to your GitHub repository in a browser.
2. Click the Actions tab.
3. You will see a workflow run called "Build and Push Docker Image" in progress.
4. Click on it, then click on the `build-and-push` job to watch live logs.

What you will see in order:
- Free up disk space (~2 min): removes Android SDK, .NET etc. to make room.
- Set up Docker Buildx (~30 sec).
- Log in to Docker Hub (~10 sec).
- Build and push (~15-25 min): the main work.
  - Pulling the CUDA base image (~5 min, done once then cached).
  - apt-get install (~2 min).
  - pip install PyTorch (~4 min).
  - pip install requirements.txt (~2 min).
  - git clone zkllm-ccs2024 (~1 min).
  - Applying patches (~10 sec).
  - Copying your scripts (~5 sec).
  - Pushing layers to Docker Hub (~3-5 min).

The final log line from the build step looks like:
```
#xx pushing layers
#xx pushed index.docker.io/YOUR_USERNAME/zkllm-smollm2@sha256:...
```

If the workflow shows a green checkmark, the image is on Docker Hub.

### If the build fails

The most common failures and their fixes:

**Error: "no space left on device"**
The free-disk-space step did not run or did not free enough space.
Fix: check that the `jlumbroso/free-disk-space@main` step appears first in
the workflow and that `large-packages: true` is set.

**Error: "unauthorized: incorrect username or password"**
The Docker Hub token is wrong or expired.
Fix: generate a new token on Docker Hub, update the `DOCKERHUB_TOKEN` secret.

**Error: "pull access denied" or "repository does not exist"**
The DOCKERHUB_USERNAME secret has a typo.
Fix: check it exactly matches your Docker Hub username (case-sensitive).

---

## 7. Verify on Docker Hub

1. Go to hub.docker.com/r/YOUR_USERNAME/zkllm-smollm2/tags
2. You should see two tags: `latest` and a long hex string (the git SHA).
3. The image size should be around 6-8 GB compressed.
4. Click the repository name → Settings → set Visibility to Public.
   This lets RunPod pull it without needing credentials.

---

## 8. Run on RunPod H200

### 8a. Deploy a pod

1. Log in at runpod.io → click Pods → Deploy.
2. Click Secure Cloud.
3. In the GPU list, find H200 SXM (80GB or 141GB).
4. For the template, choose Custom (do not use a pre-built template).
5. In the Container Image field, type:
   ```
   YOUR_USERNAME/zkllm-smollm2:latest
   ```
6. Set Container Disk to 30 GB (for the OS + compiled binaries).
7. Set Volume Disk to 100 GB, mounted at `/app/zkllm-workdir`.
   This volume persists the proof files and HuggingFace model weights
   across container restarts. The model alone is ~270 MB so 100 GB is generous.
8. Under Environment Variables, add:
   ```
   OUTPUT_DIR    /app/output
   N_LAYERS      30
   SEQ_LEN       512
   ```
   If you want to use a gated HuggingFace model later, also add:
   ```
   HF_TOKEN      hf_your_token_here
   ```
9. Under Expose Ports, you can leave this empty — this workload is not a server.
10. Click Deploy On-Demand.

Wait ~1-2 minutes for the pod to show status "Running".

### 8b. Understand what happens at startup

When the container starts, the ENTRYPOINT runs two things in sequence:

Step 1 — compile_zkllm.sh:
- Detects the H200's compute capability (sm_90).
- Patches the Makefile in /app/zkllm-ccs2024 to use /usr/local/cuda/bin/nvcc
  and sm_90 architecture.
- Runs make to compile ppgen, commit-param, rmsnorm, self-attn, ffn, skip-connection.
- Verifies all 6 binaries exist and are executable.
- This takes about 3-5 minutes on first run.

Step 2 — entrypoint.py:
- Downloads SmolLM2-135M from HuggingFace (~270 MB, cached to /app/hf_cache).
- Generates public parameters (ppgen) for each weight shape.
- Runs the 30-layer proof pipeline: for each layer, commits weights then proves.
- Evaluates C4 perplexity (original and quantized).
- Writes final_metrics.txt, per_layer_metrics.csv, metrics.json to /app/output.

### 8c. Watch the logs

On the RunPod pod card, click Connect → Logs (or use the web terminal).

You will see:
```
[INFO] GPU   : NVIDIA H200
[INFO] CUDA  : 12.1
[INFO] CWD set to: /app/zkllm-ccs2024
[compile] Detected GPU architecture: sm_90
[compile] Building with -j48 ...
[compile] All binaries built successfully.
[INFO] Loading HuggingFaceTB/SmolLM2-135M ...
[INFO] Model loaded in 8.2s  layers=30 hidden=576 heads=9 inter=1536
[INFO] Checking / generating public parameters ...
Sweeping 30 layers at SEQ_LEN=512

[L00] commit= 12.3s (  45.1 MB)  prove= 38.4s  gpu=2.14 GB  total= 52.1s
[L01] commit=  9.8s (  45.1 MB)  prove= 37.9s  gpu=2.14 GB  total= 48.2s
...
```

### 8d. Retrieve your results

Option 1 — Web terminal (easiest):
```bash
# In RunPod web terminal
cat /app/output/final_metrics.txt
```

Option 2 — Copy to your laptop via scp (if you added an SSH key to RunPod):
```bash
scp -i ~/.ssh/your_runpod_key root@RUNPOD_IP:/app/output/final_metrics.txt ./
```

Option 3 — Upload to S3/GCS: add a step at the end of entrypoint.py.

### 8e. Terminate the pod when done

On the RunPod dashboard, click the red X next to your pod and confirm termination.
H200 pods are expensive — do not leave them running.

The volume (/app/zkllm-workdir) is preserved after pod termination and can be
reattached to a new pod. This means public parameters and model weights
do not need to be regenerated on the next run.

---

## 9. Environment variables reference

These can all be overridden at `docker run -e` or in RunPod's environment
variables section without rebuilding the image.

| Variable    | Default                          | Notes                               |
|-------------|----------------------------------|-------------------------------------|
| MODEL_CARD  | HuggingFaceTB/SmolLM2-135M      | Any HuggingFace model card path     |
| N_LAYERS    | 30                               | Number of transformer layers to run |
| SEQ_LEN     | 512                              | Sequence length (affects proof size)|
| OUTPUT_DIR  | /app/output                      | Where metrics files are written     |
| WORKDIR     | /app/zkllm-workdir               | Working directory for proof files   |
| ZKLLM_REPO  | /app/zkllm-ccs2024               | Path to the compiled zkLLM repo     |
| HF_TOKEN    | (unset)                          | HuggingFace token for gated models  |

---

## 10. Rebuilding the image

After you make any code change:

```bash
git add .
git commit -m "describe your change"
git push
```

GitHub Actions triggers automatically. The new build reuses cached layers
(CUDA base, apt, pip) so it typically finishes in 5-8 minutes instead of 20+.

To force a full rebuild (e.g. after changing requirements.txt):
Just push — Docker's layer cache handles this correctly.
If you need to clear the GitHub Actions cache: go to Actions → Caches and
delete the relevant entry, then push again.

---

## 11. Moving to Phala (after H200 validation)

Phala requires:
1. A public Docker image (you already have this).
2. The image must be linux/amd64 (already set in the workflow).
3. A docker-compose.yml in your repository (already created).

Edit docker-compose.yml:
- Replace `YOUR_DOCKERHUB_USERNAME` with your actual Docker Hub username.
- Phala's TEE nodes run on specific Intel TDX or AMD SEV hardware — check
  Phala's current documentation for any additional base image requirements
  before deploying there.

The docker-compose.yml in this repo already has the correct structure for Phala.
