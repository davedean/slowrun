#!/bin/bash
# NCA Pre-Pre-Training Pipeline for Slowrun
#
# Run this BEFORE the main slowrun training to pre-train trunk weights
# on synthetic NCA data. Then pass the checkpoint to train.py.
#
# Usage:
#   cd nca/
#   bash run.sh              # generates data + trains + transfers
#   cd ..
#   torchrun --standalone --nproc_per_node=8 train.py --pretrained-checkpoint nca/checkpoints/transferred.pt
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

# Step 2: Pre-train on NCA data
CONFIG=${NCA_CONFIG:-full}
if [ ! -f checkpoints/nca_best.pt ]; then
    echo ">>> Step 2: Pre-training on NCA data (config: $CONFIG)..."

    # Detect number of GPUs
    NUM_GPUS=0
    if command -v nvidia-smi &>/dev/null; then
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    fi

    TRAIN_ARGS=(
        --data-dir ./data
        --output-dir ./checkpoints
        --config "$CONFIG"
        ${NCA_DEVICE:+--device "$NCA_DEVICE"}
        ${NCA_EPOCHS:+--epochs "$NCA_EPOCHS"}
    )

    if [ "$NUM_GPUS" -gt 1 ] && [ -z "$NCA_DEVICE" ]; then
        echo "    Detected $NUM_GPUS GPUs, using torchrun DDP"
        $PYTHON -m torch.distributed.run \
            --standalone --nproc_per_node="$NUM_GPUS" \
            pretrain.py "${TRAIN_ARGS[@]}"
    else
        echo "    Single GPU/CPU mode"
        $PYTHON pretrain.py "${TRAIN_ARGS[@]}"
    fi
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
