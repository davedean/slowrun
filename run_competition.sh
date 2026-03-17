#!/bin/bash
# Competition run: NCA pre-pretrained tiny track
# Usage: bash run_competition.sh [wandb_api_key]
set -e

WANDB_KEY=${1:-""}

echo "=== Slowrun Tiny Track — NCA Pre-Pre-Training ==="
echo ""

# Setup
cd /root
if [ ! -d slowrun ]; then
    git clone -b nca-pre-pretraining https://github.com/davedean/slowrun.git
fi
cd slowrun
pip install -r requirements.txt 2>&1 | tail -1

# Data
if [ ! -f fineweb_data/fineweb_train.pt ]; then
    python3 prepare_data.py
else
    echo "FineWeb data already exists"
fi

# Wandb
if [ -n "$WANDB_KEY" ]; then
    wandb login "$WANDB_KEY"
    export WANDB_MODE=online
else
    echo "No wandb key — logging disabled"
    export WANDB_MODE=disabled
fi

# Go
echo ""
echo "=== Starting training ==="
PYTHONUNBUFFERED=1 NCA_TOKENS=2000000 NCA_EPOCHS=1 \
    torchrun --standalone --nproc_per_node=8 tiny/train.py
