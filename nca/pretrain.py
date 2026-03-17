"""
NCA pre-pre-training: train the slowrun architecture on NCA data.

Supports DDP multi-GPU training via torchrun for fast pretraining.

Usage (single GPU):
    python pretrain.py --data-dir ./data --config tiny

Usage (multi-GPU):
    torchrun --standalone --nproc_per_node=8 pretrain.py --data-dir ./data --config tiny-track
"""

import argparse
import contextlib
import copy
import math
import os
import time

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig

# ── Configs ──────────────────────────────────────────────────────────────────

CONFIGS = {
    "tiny": dict(
        n_layer=4, n_head=4, n_kv_head=4, n_embd=256,
        batch_size=64, grad_accum=1, lr=1e-3, epochs=3,
    ),
    "medium": dict(
        n_layer=8, n_head=8, n_kv_head=8, n_embd=512,
        batch_size=4, grad_accum=1, lr=3e-4, epochs=3,
    ),
    "small": dict(
        n_layer=12, n_head=8, n_kv_head=8, n_embd=768,
        batch_size=32, grad_accum=2, lr=3e-4, epochs=3,
    ),
    "tiny-track": dict(
        n_layer=16, n_head=8, n_kv_head=8, n_embd=1024,
        batch_size=4, grad_accum=1, lr=2e-4, epochs=3,
    ),
    "full": dict(
        n_layer=30, n_head=14, n_kv_head=14, n_embd=1792,
        batch_size=8, grad_accum=8, lr=1e-4, epochs=3,
    ),
}


# ── DDP helpers ──────────────────────────────────────────────────────────────

def setup_distributed():
    """Initialize DDP if launched via torchrun."""
    if "RANK" not in os.environ:
        return 0, 0, 1
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ── Data loading ─────────────────────────────────────────────────────────────

def load_nca_data(data_dir, split="train"):
    path = os.path.join(data_dir, f"nca_{split}.pt")
    data = torch.load(path, weights_only=True)
    tokens = data["tokens"].long()
    return tokens


def get_batch(tokens, batch_size, seq_len, device, rng):
    """Sample a random batch of sequences."""
    n_seqs = tokens.shape[0]
    idx = torch.randint(0, n_seqs, (batch_size,), generator=rng)
    seqs = tokens[idx]
    x = seqs[:, :seq_len].to(device)
    y = seqs[:, 1:seq_len+1].to(device)
    return x, y


# ── Learning rate schedule ───────────────────────────────────────────────────

def get_lr(step, total_steps, max_lr, warmup_frac=0.05, min_lr_frac=0.1):
    warmup_steps = int(total_steps * warmup_frac)
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 *
                     (1 + math.cos(math.pi * progress)))


# ── Training loop ────────────────────────────────────────────────────────────

def train(args):
    rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0
    ddp = world_size > 1

    cfg = CONFIGS[args.config]
    seq_len = args.seq_len

    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    device_type = device.split(":")[0]

    if is_main:
        print(f"Config: {args.config}")
        print(f"Device: {device} (world_size={world_size})")

    # Load data
    train_tokens = load_nca_data(args.data_dir, "train")
    val_tokens = load_nca_data(args.data_dir, "val")

    if is_main:
        print(f"Loaded train: {train_tokens.shape}, val: {val_tokens.shape}")

    # Determine vocab size from data (or use override)
    max_token = max(train_tokens.max().item(), val_tokens.max().item())
    vocab_size = getattr(args, 'vocab_size', None) or (max_token + 1)
    if is_main:
        print(f"Vocab size: {vocab_size} (max token in data: {max_token})")

    # Create model
    model_config = GPTConfig(
        vocab_size=vocab_size,
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_kv_head=cfg["n_kv_head"],
        n_embd=cfg["n_embd"],
        sequence_len=seq_len,
        dropout=0.0,
    )
    model = GPT(model_config)
    model.init_weights()
    model = model.to(device)

    # DDP wrap
    if ddp:
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    raw_model = model.module if ddp else model

    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Model params: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  betas=(0.9, 0.95), weight_decay=0.1)

    # Training setup — batch_size is per-GPU, effective = bs * accum * world
    batch_size = cfg["batch_size"]
    grad_accum = cfg["grad_accum"]
    effective_batch = batch_size * grad_accum * world_size
    tokens_per_step = effective_batch * seq_len

    n_train_tokens = train_tokens.shape[0] * seq_len
    steps_per_epoch = n_train_tokens // tokens_per_step
    total_steps = steps_per_epoch * cfg["epochs"]
    skip_eval = getattr(args, 'skip_eval', False)
    eval_interval = max(1, steps_per_epoch // 2) if not skip_eval else total_steps + 1

    if is_main:
        print(f"Batch: {batch_size}/gpu x {grad_accum} accum x {world_size} gpus"
              f" = {effective_batch} effective")
        print(f"Steps/epoch: {steps_per_epoch}, total: {total_steps}")
        print(f"Eval every {eval_interval} steps")

    # bf16 autocast
    use_amp = device_type != "cpu"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    rng = torch.Generator()
    rng.manual_seed(getattr(args, 'seed', 42) + rank)
    val_rng = torch.Generator()
    val_rng.manual_seed(123)

    best_val_loss = float("inf")
    t0 = time.time()

    # Periodic checkpoint saving
    save_every_tokens = getattr(args, 'save_every_tokens', 0)
    next_save_at = save_every_tokens if save_every_tokens > 0 else float("inf")

    for step in range(total_steps):
        lr = get_lr(step, total_steps, cfg["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        optimizer.zero_grad()
        train_loss_accum = 0.0

        for micro in range(grad_accum):
            x, y = get_batch(train_tokens, batch_size, seq_len, device, rng)
            ctx = model.no_sync() if (ddp and micro < grad_accum - 1) else \
                contextlib.nullcontext()
            with ctx:
                with torch.autocast(device_type=device_type,
                                    dtype=amp_dtype, enabled=use_amp):
                    loss = model(x, y) / grad_accum
                loss.backward()
            train_loss_accum += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Periodic token-based checkpoint (rank 0 only)
        tokens_so_far = (step + 1) * tokens_per_step
        if is_main and tokens_so_far >= next_save_at:
            tok_m = int(next_save_at) // 1_000_000
            ckpt_path = os.path.join(args.output_dir, f"nca_at_{tok_m}M.pt")
            torch.save({
                "model_state_dict": raw_model.state_dict(),
                "config": model_config,
                "step": step,
                "tokens_seen": tokens_so_far,
                "val_loss": best_val_loss,
            }, ckpt_path)
            print(f"  >> saved periodic checkpoint at ~{tok_m}M tokens: {ckpt_path}")
            next_save_at += save_every_tokens

        # Logging
        if is_main and step % 50 == 0:
            elapsed = time.time() - t0
            tok_per_sec = (step + 1) * tokens_per_step / elapsed
            print(f"step {step:>5d}/{total_steps} | "
                  f"loss {train_loss_accum:.4f} | "
                  f"lr {lr:.2e} | {tok_per_sec:,.0f} tok/s")

        # Evaluation (rank 0 only, skipped if --skip-eval)
        if is_main and not skip_eval and (step % eval_interval == 0 or step == total_steps - 1):
            model.eval()
            val_losses = []
            val_bs = min(batch_size, val_tokens.shape[0])
            n_val_batches = max(1, min(20, val_tokens.shape[0] // val_bs))
            with torch.no_grad():
                for _ in range(n_val_batches):
                    x, y = get_batch(val_tokens, val_bs, seq_len,
                                     device, val_rng)
                    with torch.autocast(device_type=device_type,
                                        dtype=amp_dtype, enabled=use_amp):
                        vl = raw_model(x, y)
                    val_losses.append(vl.item())
            val_loss = sum(val_losses) / len(val_losses)
            print(f"  >> val_loss: {val_loss:.4f}"
                  f" (best: {best_val_loss:.4f})")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(args.output_dir, "nca_best.pt")
                torch.save({
                    "model_state_dict": raw_model.state_dict(),
                    "config": model_config,
                    "step": step,
                    "val_loss": val_loss,
                }, ckpt_path)
                print(f"  >> saved best checkpoint: {ckpt_path}")

    # Save final checkpoint (also as nca_best.pt if no eval was run)
    if is_main:
        state = raw_model.state_dict()
        ckpt = {
            "model_state_dict": state,
            "config": model_config,
            "step": total_steps,
            "val_loss": best_val_loss,
        }
        final_path = os.path.join(args.output_dir, "nca_final.pt")
        torch.save(ckpt, final_path)
        if best_val_loss == float("inf"):
            # No eval was run — save as best too
            torch.save(ckpt, os.path.join(args.output_dir, "nca_best.pt"))

        elapsed = time.time() - t0
        print(f"\nTraining complete in {elapsed:.1f}s")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Saved: {final_path}")

    cleanup_distributed()
    return best_val_loss


def train_from_config(data_dir, output_dir, config_name="tiny", device="cuda",
                      epochs=None, lr=None, batch_size=None, seq_len=1024,
                      save_every_tokens=0, vocab_size=None, seed=42):
    """Callable entry point for sweep scripts. Deep-copies config to avoid mutation."""
    saved = copy.deepcopy(CONFIGS[config_name])
    try:
        if epochs is not None:
            CONFIGS[config_name]["epochs"] = epochs
        if lr is not None:
            CONFIGS[config_name]["lr"] = lr
        if batch_size is not None:
            CONFIGS[config_name]["batch_size"] = batch_size
        os.makedirs(output_dir, exist_ok=True)
        args = argparse.Namespace(
            data_dir=data_dir, output_dir=output_dir, config=config_name,
            device=device, seq_len=seq_len, save_every_tokens=save_every_tokens,
            vocab_size=vocab_size, seed=seed,
        )
        return train(args)
    finally:
        CONFIGS[config_name] = saved


def main():
    parser = argparse.ArgumentParser(description="NCA pre-pre-training")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./nca_checkpoints")
    parser.add_argument("--config", type=str, default="tiny",
                        choices=list(CONFIGS.keys()))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--save-every-tokens", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip validation during training (faster)")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile (slower startup, faster training)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.epochs is not None:
        CONFIGS[args.config]["epochs"] = args.epochs
    if args.lr is not None:
        CONFIGS[args.config]["lr"] = args.lr
    if args.batch_size is not None:
        CONFIGS[args.config]["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        CONFIGS[args.config]["grad_accum"] = args.grad_accum

    train(args)


if __name__ == "__main__":
    main()
