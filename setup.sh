#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# CogNet-1B Quick Start Setup
# ═══════════════════════════════════════════════════════════════════
set -e

echo "=========================================="
echo "CogNet-1B Setup"
echo "=========================================="

# 1. Install dependencies
echo "[1/3] Installing dependencies..."
pip install torch --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || pip install torch
pip install datasets huggingface_hub tokenizers safetensors

# 2. Set HF token
echo ""
echo "[2/3] Setting up HuggingFace token..."
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set!"
    echo "Set it with: export HF_TOKEN=hf_your_token_here"
    echo "Or login with: huggingface-cli login"
fi

# 3. Create directories
echo ""
echo "[3/3] Creating directories..."
mkdir -p data_1b checkpoints_1b

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "Quick commands:"
echo "  # Prepare datasets (downloads + tokenizes everything)"
echo "  python train_ultra.py --skip-data-prep=False --max-steps 0"
echo ""
echo "  # Train on single GPU"
echo "  python train_ultra.py --max-steps 100000"
echo ""
echo "  # Train on multi-GPU"
echo "  torchrun --nproc_per_node=4 train_ultra.py --use-fsdp --max-steps 100000"
echo ""
echo "  # Train on ACIL/RunPod"
echo "  export HF_TOKEN=your_token"
echo "  sbatch acil_submit.sh"
echo "=========================================="
