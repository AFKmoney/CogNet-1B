#!/bin/bash
#SBATCH --job-name=cognet-1b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=logs/cognet-%j.out
#SBATCH --error=logs/cognet-%j.err

# ═══════════════════════════════════════════════════════════════════
# CogNet-1B Training V2 — ACIL Cluster (MAXIMUM SPEED)
# Architecture: 16 blocks, 8 channels, channel_dim=384, ff_dim=8192
# Datasets: 7 HF datasets + AICL repo + scripts + synthetic
# Optimizations: 12+1 (BF16, FSDP, compile, SDPA, prefetch, etc.)
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

echo "=========================================="
echo "CogNet-1B Training V2 — ACIL Cluster"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

# ── Environment ──
export HF_TOKEN="${HF_TOKEN:-}"
export COGNET_WORKSPACE="${COGNET_WORKSPACE:-/workspace/CogNet}"
export AICL_REPEAT=10

# CUDA optimizations
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Better overlap for NCCL
export NCCL_P2P_LEVEL=NVL            # NVLink optimization if available
export TORCH_NCCL_AVOID_RECORD_STREAMS=1  # Reduce NCCL memory overhead

# ── Create directories ──
mkdir -p logs checkpoints_1b data_1b

# ── Config (matches HuggingFace CogNet-1B) ──
MODEL_SIZE="1b"
BATCH_SIZE=4
GRAD_ACCUM=8
SEQ_LEN=512
MAX_LR=1e-4
MIN_LR=1e-5
WARMUP_STEPS=2000
MAX_STEPS=100000
CKPT_DIR="./checkpoints_1b"

NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}

echo ""
echo "Config:"
echo "  Model: $MODEL_SIZE (16 blocks, 384 channel, 8192 ff)"
echo "  Vocab: 136 (CharTokenizer)"
echo "  Seq len: $SEQ_LEN"
echo "  Batch/GPU: $BATCH_SIZE"
echo "  Grad accum: $GRAD_ACCUM"
echo "  Effective batch: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "  LR: $MAX_LR -> $MIN_LR"
echo "  Steps: $MAX_STEPS"
echo "  HF token: ${HF_TOKEN:+SET}${HF_TOKEN:-NOT SET}"
echo "  GPUs: $NUM_GPUS"
echo "  All optimizations: ON"
echo ""

# ── Step 1: Data Preparation (run once) ──
MERGED_DATA="./data_1b/train_merged.pt"
if [ ! -f "$MERGED_DATA" ]; then
    echo "[Step 1] Preparing datasets (HF + AICL + synthetic)..."
    python train_ultra.py --max-steps 0 --skip-data-prep
else
    echo "[Step 1] Datasets already prepared"
fi

# ── Step 2: Train with ALL optimizations ──
echo ""
echo "[Step 2] Starting training with MAXIMUM SPEED optimizations..."
echo ""

# Common optimization flags
OPTIM_FLAGS="--bf16 --compile --cuda-prefetch --seq-warmup --async-ckpt"

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun \
        --standalone \
        --nproc_per_node="$NUM_GPUS" \
        train_ultra.py \
        --model-size "$MODEL_SIZE" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --seq-len "$SEQ_LEN" \
        --max-lr "$MAX_LR" \
        --min-lr "$MIN_LR" \
        --warmup-steps "$WARMUP_STEPS" \
        --max-steps "$MAX_STEPS" \
        --ckpt-dir "$CKPT_DIR" \
        --use-fsdp \
        $OPTIM_FLAGS \
        --skip-data-prep \
        --save-every 2000 \
        --eval-every 500
else
    python train_ultra.py \
        --model-size "$MODEL_SIZE" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --seq-len "$SEQ_LEN" \
        --max-lr "$MAX_LR" \
        --min-lr "$MIN_LR" \
        --warmup-steps "$WARMUP_STEPS" \
        --max-steps "$MAX_STEPS" \
        --ckpt-dir "$CKPT_DIR" \
        $OPTIM_FLAGS \
        --compile-step \
        --skip-data-prep \
        --save-every 2000 \
        --eval-every 500
fi

echo ""
echo "=========================================="
echo "Training complete!"
echo "End: $(date)"
echo "=========================================="
