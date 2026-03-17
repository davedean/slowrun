"""
NCA data generation with GPT-2 tokenization — renders grids as text.

Instead of patch-based 10K vocab, renders NCA grids as digit strings
and tokenizes with GPT-2 BPE. This means no weight transfer is needed —
the entire model (including embeddings) transfers directly.
"""

import os
import time

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from generate_data_gpu import NCASimulatorGPU, _compute_entropy_gpu


def grid_entropy_filter(snapshots, num_colors=10, lower=1.0, upper=2.3):
    """Filter rules by cell-value entropy across the trajectory.

    snapshots: (B, T, H, W) int tensor on GPU
    Returns: boolean mask (B,)

    Entropy is computed on raw grid cell values (0-9), not tokens.
    Calibrated: boring grids (all same) → ~0, pure noise → log2(10)≈3.32
    [1.0, 2.3] gives ~4% acceptance, matching gzip filter selectivity.
    """
    B, T, H, W = snapshots.shape
    flat = snapshots.reshape(B, -1).long()  # (B, T*H*W)
    L = flat.shape[1]
    counts = torch.zeros(B, num_colors, device=flat.device)
    counts.scatter_add_(1, flat, torch.ones_like(flat, dtype=torch.float))
    probs = counts / L
    log_probs = torch.where(probs > 0, torch.log2(probs), torch.zeros_like(probs))
    entropy = -(probs * log_probs).sum(dim=1)
    return (entropy >= lower) & (entropy <= upper)


def grids_to_text_tokens(snapshots, enc):
    """Convert batched grid trajectories to GPT-2 token sequences.

    snapshots: (B, T, H, W) int tensor on GPU (only accepted rules)
    Returns: list of 1D numpy arrays of GPT-2 token IDs
    """
    B, T, H, W = snapshots.shape
    grids = snapshots.cpu().numpy()

    all_tokens = []
    for b in range(B):
        parts = []
        for t in range(T):
            grid = grids[b, t]
            text = '\n'.join(' '.join(str(c) for c in row) for row in grid)
            parts.append(text)
        full_text = '\n---\n'.join(parts)
        tokens = enc.encode(full_text)
        all_tokens.append(np.array(tokens, dtype=np.int32))

    return all_tokens


def generate_nca_data_gpt2(num_tokens, grid_size=12, num_colors=10,
                           dt=2, steps_per_rule=10, start_step=0,
                           seed=42, seq_len=1025, val_fraction=0.05,
                           device="cuda", batch_size=256):
    """Generate NCA data tokenized with GPT-2 BPE."""
    torch.manual_seed(seed)

    sim = NCASimulatorGPU(grid_size=grid_size, num_colors=num_colors, device=device)
    enc = tiktoken.get_encoding('gpt2')

    # ~158 GPT-2 tokens per grid, 10 grids per rule = ~1580 tokens per rule
    est_tokens_per_rule = 1580

    print(f"Grid: {grid_size}x{grid_size}, colors: {num_colors}")
    print(f"Tokenizer: GPT-2 BPE (vocab 50257)")
    print(f"~{est_tokens_per_rule} tokens per rule")
    print(f"Target: {num_tokens:,} tokens")
    print(f"GPU batch size: {batch_size}")

    all_tokens = []
    total_generated = 0
    total_rules = 0
    t0 = time.time()

    total_accepted = 0
    total_rejected = 0

    while total_generated < num_tokens:
        # Generate batch of rules on GPU
        params = sim.random_params(batch_size)
        snapshots = sim.rollout(params, num_steps=steps_per_rule,
                                dt=dt, start_step=start_step)

        # Filter on raw grid entropy (GPU)
        mask = grid_entropy_filter(snapshots, num_colors=num_colors)
        accepted_snaps = snapshots[mask]
        total_accepted += mask.sum().item()
        total_rejected += (~mask).sum().item()

        if accepted_snaps.shape[0] == 0:
            continue

        # Convert accepted rules to GPT-2 tokens (CPU-bound text rendering)
        token_seqs = grids_to_text_tokens(accepted_snaps, enc)

        for seq in token_seqs:
            all_tokens.append(seq)
            total_generated += len(seq)
            total_rules += 1

        elapsed = time.time() - t0
        rate = total_generated / elapsed if elapsed > 0 else 0
        accept_rate = total_accepted / max(1, total_accepted + total_rejected)

        if total_rules % (batch_size * 2) < batch_size or total_generated >= num_tokens:
            print(f"  {total_generated:>12,} / {num_tokens:,} tokens | "
                  f"accept {accept_rate:.1%} | {rate:,.0f} tok/s | "
                  f"{elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"\nDone: {total_generated:,} tokens in {elapsed:.1f}s")

    # Pack into sequences
    flat = np.concatenate(all_tokens)
    num_seqs = len(flat) // seq_len
    sequences = flat[:num_seqs * seq_len].reshape(num_seqs, seq_len)
    print(f"Packed into {num_seqs:,} sequences of length {seq_len}")

    # Split train/val
    rng = np.random.default_rng(seed)
    n_val = max(1, int(num_seqs * val_fraction))
    n_train = num_seqs - n_val
    indices = rng.permutation(num_seqs)
    train_seqs = sequences[indices[:n_train]]
    val_seqs = sequences[indices[n_train:]]

    train_tokens = torch.tensor(train_seqs, dtype=torch.int32)
    val_tokens = torch.tensor(val_seqs, dtype=torch.int32)
    print(f"Train: {train_tokens.shape}, Val: {val_tokens.shape}")

    return train_tokens, val_tokens


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate NCA data with GPT-2 vocab")
    parser.add_argument("--output-dir", type=str, default="./data")
    parser.add_argument("--num-tokens", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train, val = generate_nca_data_gpt2(
        num_tokens=args.num_tokens, batch_size=args.batch_size,
        seed=args.seed, device=args.device,
    )
    torch.save({"tokens": train}, os.path.join(args.output_dir, "nca_train.pt"))
    torch.save({"tokens": val}, os.path.join(args.output_dir, "nca_val.pt"))
    print(f"Saved to {args.output_dir}/")
