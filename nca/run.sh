#!/bin/bash
# NCA Pre-Pre-Training Pipeline for Slowrun
#
# Run this BEFORE the main slowrun training to pre-train trunk weights
# on synthetic NCA data. Then pass the checkpoint to train.py.
#
# Usage:
#   cd nca/
#   bash run.sh
#   cd ..
#   torchrun --standalone --nproc_per_node=8 train.py --pretrained-checkpoint nca/checkpoints/transferred.pt
#
# Environment variables:
#   NCA_CONFIG   - pretrain config: tiny, small, full (default: full)
#   NCA_DEVICE   - device override (default: auto-detect)
#   NCA_EPOCHS   - override number of epochs
#   NCA_SEED     - training seed (default: 42)
#   NUM_WORKERS  - data generation workers (default: 8)
#   PYTHON       - python binary (default: python3)
set -e

PYTHON=${PYTHON:-python3}
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=== NCA Pre-Pre-Training Pipeline ==="
echo ""

# Step 1: Generate NCA data (CPU-only, can run on any machine)
if [ ! -f data/nca_train.pt ]; then
    echo ">>> Step 1: Generating NCA data..."
    $PYTHON generate_data.py \
        --output-dir ./data \
        --num-tokens 164000000 \
        --num-workers "${NUM_WORKERS:-8}"
else
    echo ">>> Step 1: NCA data already exists, skipping"
fi
echo ""

# Step 2: Pre-train on NCA data (single GPU)
CONFIG=${NCA_CONFIG:-full}
if [ ! -f checkpoints/nca_best.pt ]; then
    echo ">>> Step 2: Pre-training on NCA data (config: $CONFIG)..."
    $PYTHON pretrain.py \
        --data-dir ./data \
        --output-dir ./checkpoints \
        --config "$CONFIG" \
        ${NCA_DEVICE:+--device "$NCA_DEVICE"} \
        ${NCA_EPOCHS:+--epochs "$NCA_EPOCHS"} \
        ${NCA_SEED:+--seed "$NCA_SEED"}
else
    echo ">>> Step 2: NCA checkpoint already exists, skipping"
fi
echo ""

# Step 3: Transfer weights to slowrun vocab
if [ ! -f checkpoints/transferred.pt ]; then
    echo ">>> Step 3: Transferring weights..."
    $PYTHON transfer.py \
        --nca-checkpoint ./checkpoints/nca_best.pt \
        --output ./checkpoints/transferred.pt \
        --target-vocab-size 50257
else
    echo ">>> Step 3: Transferred checkpoint already exists, skipping"
fi

echo ""
echo "=== Done! ==="
echo "Run slowrun with:"
echo "  torchrun --standalone --nproc_per_node=8 train.py --pretrained-checkpoint nca/checkpoints/transferred.pt"
