#!/bin/bash
# NCA Pre-Pre-Training Pipeline for Slowrun
#
# Two modes:
#   GPT-2 vocab (default): generates NCA data tokenized with GPT-2 BPE,
#     trains directly — no weight transfer needed, all weights including
#     embeddings carry over.
#   10K vocab (legacy): patch-based 10K vocab, requires weight transfer.
#
# Usage:
#   cd nca/
#   bash run.sh
#   cd ..
#   torchrun --standalone --nproc_per_node=8 train.py --pretrained-checkpoint nca/checkpoints/nca_best.pt
#
# Environment variables:
#   NCA_CONFIG   - pretrain config: tiny, medium, small, full (default: full)
#   NCA_DEVICE   - device override (default: auto-detect)
#   NCA_EPOCHS   - override number of epochs
#   NCA_TOKENS   - number of NCA tokens to generate (default: 10000000)
#   NCA_SEED     - training seed (default: 42)
#   NCA_MODE     - "gpt2" (default) or "10k" (legacy patch vocab)
#   NUM_WORKERS  - data generation workers for 10k mode (default: 8)
#   PYTHON       - python binary (default: python3)
set -e

PYTHON=${PYTHON:-python3}
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

MODE=${NCA_MODE:-gpt2}
CONFIG=${NCA_CONFIG:-full}
TOKENS=${NCA_TOKENS:-10000000}

echo "=== NCA Pre-Pre-Training Pipeline (mode: $MODE) ==="
echo ""

if [ "$MODE" = "gpt2" ]; then
    # ── GPT-2 vocab pipeline (no transfer needed) ──────────────
    if [ ! -f data/nca_train.pt ]; then
        echo ">>> Step 1: Generating NCA data (GPT-2 vocab, GPU)..."
        $PYTHON generate_data_gpt2.py \
            --num-tokens "$TOKENS" \
            --output-dir ./data \
            ${NCA_SEED:+--seed "$NCA_SEED"}
    else
        echo ">>> Step 1: NCA data already exists, skipping"
    fi
    echo ""

    if [ ! -f checkpoints/nca_best.pt ]; then
        # Detect GPU count for DDP
        NGPU=$(${PYTHON} -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
        echo ">>> Step 2: Pre-training on NCA data (config: $CONFIG, GPT-2 vocab, ${NGPU} GPUs)..."
        if [ "$NGPU" -gt 1 ]; then
            # Use different master port and clear parent DDP env to avoid conflicts
            unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK LOCAL_WORLD_SIZE ROLE_RANK TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS TORCHELASTIC_RUN_ID 2>/dev/null
            torchrun --standalone --nproc_per_node="$NGPU" --master_port 29501 pretrain.py \
                --data-dir ./data \
                --output-dir ./checkpoints \
                --config "$CONFIG" \
                --vocab-size 50257 \
                ${NCA_EPOCHS:+--epochs "$NCA_EPOCHS"} \
                ${NCA_SEED:+--seed "$NCA_SEED"}
        else
            $PYTHON pretrain.py \
                --data-dir ./data \
                --output-dir ./checkpoints \
                --config "$CONFIG" \
                --vocab-size 50257 \
                ${NCA_DEVICE:+--device "$NCA_DEVICE"} \
                ${NCA_EPOCHS:+--epochs "$NCA_EPOCHS"} \
                ${NCA_SEED:+--seed "$NCA_SEED"}
        fi
    else
        echo ">>> Step 2: NCA checkpoint already exists, skipping"
    fi

    echo ""
    echo "=== Done! ==="
    echo "Checkpoint ready at nca/checkpoints/nca_best.pt (no transfer needed)"
    echo "Run slowrun with:"
    echo "  torchrun --standalone --nproc_per_node=8 train.py --pretrained-checkpoint nca/checkpoints/nca_best.pt"

else
    # ── Legacy 10K vocab pipeline (with transfer) ──────────────
    if [ ! -f data/nca_train.pt ]; then
        echo ">>> Step 1: Generating NCA data (10K vocab, CPU)..."
        $PYTHON generate_data.py \
            --output-dir ./data \
            --num-tokens "$TOKENS" \
            --num-workers "${NUM_WORKERS:-8}"
    else
        echo ">>> Step 1: NCA data already exists, skipping"
    fi
    echo ""

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
fi
