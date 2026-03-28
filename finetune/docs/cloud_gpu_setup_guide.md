# Cloud GPU Setup Guide

Step-by-step instructions for provisioning a cloud GPU, uploading code, launching training, and downloading results — no deep ML expertise required.

---

## Overview

| Step | Action | Time |
|------|--------|------|
| 1 | Choose cloud provider and instance | 5 min |
| 2 | Provision GPU instance | 5-10 min |
| 3 | Upload code and data scripts | 10 min |
| 4 | Install dependencies | 5-10 min |
| 5 | Download datasets | 30-120 min |
| 6 | Launch training | Hours (see estimates) |
| 7 | Download model weights | 30-60 min |

---

## Recommended Providers

### Option A: RunPod (Recommended for beginners)

**Website:** https://www.runpod.io

**Why RunPod:**
- Simple web UI, no CLI required for provisioning
- Pay-per-second billing (no wasted idle time)
- Pre-built PyTorch images (no CUDA setup)
- Persistent storage volumes (survive pod restarts)

**Cost estimates (as of early 2026):**

| GPU | VRAM | On-Demand | Spot | Best For |
|-----|------|-----------|------|----------|
| A100 SXM 80GB | 80GB | ~$2.49/hr | ~$1.49/hr | QLoRA training (recommended) |
| H100 SXM 80GB | 80GB | ~$3.99/hr | ~$2.49/hr | Fastest training |
| A100 PCIe 80GB | 80GB | ~$1.99/hr | ~$1.29/hr | Budget option |

**Recommended:** A100 SXM 80GB (Spot) for cost-effectiveness. Budget ~$50-$150 per domain depending on dataset size.

---

### Option B: Lambda Labs

**Website:** https://lambdalabs.com/service/gpu-cloud

**Why Lambda Labs:**
- Fixed hourly pricing (no auction, more predictable)
- SSH access from day 1
- Good for longer training runs

**Cost estimates:**

| GPU | VRAM | Price/hr |
|-----|------|----------|
| A100 40GB (×8) | 320GB total | ~$10.00/hr |
| H100 80GB (×8) | 640GB total | ~$24.80/hr |
| A10 24GB | 24GB | ~$0.75/hr (too small for 72B) |

**Note:** Lambda is better for multi-GPU runs. For single-GPU QLoRA, RunPod is easier.

---

## Step-by-Step: RunPod

### 1. Create Account

1. Go to https://runpod.io → Sign Up
2. Add payment method (credit card or crypto)
3. Add billing credits ($50–$200 recommended)

### 2. Create a Storage Volume

A persistent volume stores your data between pod restarts (important — pods can be interrupted).

1. Click **Storage** in the left sidebar
2. Click **New Volume**
3. Name: `llm-training`
4. Size: **500 GB** (model weights ~144GB + datasets ~50GB + outputs ~100GB)
5. Region: Choose the same region as your planned GPU
6. Click **Create Volume** (~$0.07/GB/month)

### 3. Provision GPU Pod

1. Click **Pods** → **Deploy**
2. Search for **A100 80GB SXM** (or H100 80GB)
3. Select **Secure Cloud** or **Community Cloud** (Community is cheaper)
4. Click **Customize Deployment**:
   - **Container Image:** `runpod/pytorch:2.2.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
   - **Container Disk:** 50 GB
   - **Volume Mount:** Attach your `llm-training` volume → `/workspace`
   - **Expose HTTP Ports:** (leave default)
   - **Environment Variables:** (optional — add HF_TOKEN here)
5. Click **Deploy** → Pod starts in ~2-5 minutes

### 4. Connect to Pod

1. Click your pod → **Connect**
2. Click **Start Web Terminal** (JupyterLab in browser) OR
3. Use SSH: Click **SSH** to get the connection command, then:
   ```bash
   ssh root@<pod-ip> -p <port> -i ~/.ssh/id_rsa
   ```

---

## Setting Up the Environment

Run these commands once after connecting to the pod:

```bash
# Verify GPU is detected
nvidia-smi

# Navigate to persistent storage
cd /workspace

# Upload your project from Replit (see "Uploading Code" below)
# Or clone from GitHub if you have a repo:
# git clone https://github.com/your-repo/zero-base-llm .

# Install fine-tuning dependencies
pip install -r finetune/requirements_finetune.txt

# Install flash-attention (for speed; takes ~10 minutes to compile)
pip install flash-attn --no-build-isolation

# Log into Hugging Face (needed for Qwen2.5 model download)
huggingface-cli login
# Paste your HF token from https://huggingface.co/settings/tokens
```

### Uploading Code from Replit

**Method 1: Direct file transfer (easiest)**
```bash
# On your local machine or Replit shell:
rsync -avz -e "ssh -p <port>" ./finetune/ root@<pod-ip>:/workspace/finetune/

# Or use scp:
scp -P <port> -r ./finetune/ root@<pod-ip>:/workspace/finetune/
```

**Method 2: Git (recommended for version control)**
```bash
# Push to GitHub from Replit, then on the pod:
git clone https://github.com/your-username/your-repo.git /workspace/project
cd /workspace/project
```

**Method 3: RunPod file browser**
- Use the web file manager (if using JupyterLab) to drag-and-drop files

---

## Downloading Datasets

```bash
cd /workspace/finetune

# Download all domains (takes 1-2 hours; ~20-50GB total)
python scripts/download_datasets.py --domain all --output_dir ./data/processed --limit 50000

# Or download domains individually:
python scripts/download_datasets.py --domain code --output_dir ./data/processed
python scripts/download_datasets.py --domain math --output_dir ./data/processed
python scripts/download_datasets.py --domain science --output_dir ./data/processed
python scripts/download_datasets.py --domain finance --output_dir ./data/processed
python scripts/download_datasets.py --domain general --output_dir ./data/processed
```

---

## Launching Training

### Single domain (recommended to start)

```bash
cd /workspace/finetune

# Math fine-tuning (~4-5 hours on A100 80GB)
python scripts/train.py --config configs/math.yaml

# Code fine-tuning (~6-8 hours on A100 80GB)
python scripts/train.py --config configs/code.yaml
```

### Run training in the background (so it survives disconnection)

```bash
# Use tmux (pre-installed on most pods)
tmux new-session -s training

# Inside tmux, run training:
cd /workspace/finetune
python scripts/train.py --config configs/math.yaml

# Detach from tmux: Ctrl+B, then D
# Reconnect later: tmux attach -t training
```

### Monitor training

```bash
# Watch GPU utilization:
watch -n 5 nvidia-smi

# View TensorBoard logs:
tensorboard --logdir /workspace/finetune/outputs/ --port 6006 --bind_all
# Then access at: http://<pod-ip>:6006
```

### Resume from checkpoint (if pod is interrupted)

```bash
python scripts/train.py \
    --config configs/math.yaml \
    --resume_from_checkpoint ./outputs/math/checkpoint-1000
```

---

## Expected Training Times

| Domain | Dataset Size | A100 80GB | H100 80GB | Cost (A100 Spot) |
|--------|-------------|-----------|-----------|-----------------|
| Math   | ~15K rows   | 4-5 hrs   | 2-3 hrs   | ~$6-8            |
| Science| ~13K rows   | 3-4 hrs   | 2 hrs     | ~$5-6            |
| Code   | ~50K rows   | 6-8 hrs   | 4-5 hrs   | ~$9-12           |
| Finance| ~50K rows   | 6-8 hrs   | 4-5 hrs   | ~$9-12           |
| General| ~50K rows   | 8-10 hrs  | 5-6 hrs   | ~$12-15          |
| Combined| ~250K rows | 30-40 hrs | 18-25 hrs | ~$45-60          |

*Spot prices vary. Check RunPod dashboard for current rates.*

---

## Downloading Model Weights

After training completes, download the LoRA adapter (small, ~1-2GB) and optionally the merged model:

### Download LoRA adapter only (~1-2 GB, fast)

```bash
# On your local machine:
rsync -avz -e "ssh -p <port>" \
    root@<pod-ip>:/workspace/finetune/outputs/math/final_adapter/ \
    ./downloaded_adapters/math/
```

### Download merged HF model (~144 GB, slow)

```bash
# This is large — use rclone or AWS S3 for large transfers
# Option 1: rclone to Google Drive or S3
rclone copy /workspace/finetune/exports/math-merged/ gdrive:llm-exports/math/

# Option 2: direct rsync (slow for large files)
rsync -avz --progress -e "ssh -p <port>" \
    root@<pod-ip>:/workspace/finetune/exports/math-merged/ \
    ./exports/math-merged/
```

### Upload to Hugging Face Hub (recommended)

```bash
# From the pod, push directly to HF Hub for permanent storage
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='/workspace/finetune/outputs/math/final_adapter',
    repo_id='your-username/qwen2.5-72b-math-adapter',
    repo_type='model',
)
"
```

---

## Merging and Exporting (Run on Pod)

```bash
# Merge LoRA into base model and export HF + GGUF format
python scripts/merge_and_export.py \
    --adapter_dir ./outputs/math/final_adapter \
    --base_model Qwen/Qwen2.5-72B-Instruct \
    --output_dir ./exports/math-merged \
    --gguf_quant Q4_K_M

# GGUF only needs llama.cpp setup first:
git clone https://github.com/ggerganov/llama.cpp /workspace/llama.cpp
cd /workspace/llama.cpp && pip install -r requirements.txt && make
cd /workspace/finetune
```

---

## Cost Management Tips

1. **Use Spot instances:** 40-60% cheaper; save checkpoints frequently (`save_steps: 200`)
2. **Stop pod when idle:** Training 6 hours/day at $1.49/hr = ~$9/day vs. $36/day if left running
3. **Use volumes:** Keep your data on persistent storage so you don't re-download on pod restart
4. **Monitor via tmux:** Prevents accidental training termination on disconnect
5. **Download adapters first:** Adapters are ~1-2GB vs 144GB for full models — download and iterate quickly

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `CUDA out of memory` | Batch size too large | Reduce `batch_size` to 1 and increase `gradient_accumulation_steps` |
| `RuntimeError: FlashAttention` | flash-attn not installed | `pip install flash-attn --no-build-isolation` |
| `ModuleNotFoundError: bitsandbytes` | BnB not installed | `pip install bitsandbytes>=0.43.0` |
| Pod interrupted | Spot instance preempted | Resume from latest checkpoint |
| Slow download speed | HF bandwidth limits | Use HF local cache: `export HF_HOME=/workspace/hf_cache` |
| `OSError: No space left` | Disk full | Increase container disk or delete old checkpoints |
