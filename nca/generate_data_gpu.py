"""
GPU-accelerated NCA data generation — batches hundreds of rules on CUDA.

Drop-in replacement for nca_datagen.generate_nca_data().
The NCA rollout (95% of CPU datagen time) runs entirely on GPU using
batched conv2d + Gumbel-max sampling.

Usage:
    python nca_datagen_gpu.py --output-dir ./nca_data --num-tokens 164000000
"""

import argparse
import gzip
import io
import os
import time

import numpy as np
import torch
import torch.nn.functional as F


def gzip_complexity(data_bytes):
    """Gzip compression ratio — lower means more compressible."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=9) as f:
        f.write(data_bytes)
    return len(buf.getvalue()) / len(data_bytes)


# ── GPU NCA Simulator ─────────────────────────────────────────────────────────

class NCASimulatorGPU:
    """Batched discrete NCA on GPU: simulate many rules simultaneously."""

    def __init__(self, grid_size=12, num_colors=10, device="cuda"):
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.device = device

    def random_params(self, batch_size):
        """Generate random NCA params for a batch of rules."""
        C = self.num_colors
        # Conv 3x3: (batch, 4, C, 3, 3)
        w1 = torch.randn(batch_size, 4, C, 3, 3, device=self.device) * 0.5
        b1 = torch.zeros(batch_size, 4, device=self.device)
        # 1x1 conv as matmul: (batch, 16, 4)
        w2 = torch.randn(batch_size, 16, 4, device=self.device) * 0.5
        b2 = torch.zeros(batch_size, 16, device=self.device)
        # 1x1 conv as matmul: (batch, C, 16)
        w3 = torch.randn(batch_size, C, 16, device=self.device) * 0.5
        b3 = torch.zeros(batch_size, C, device=self.device)
        return w1, b1, w2, b2, w3, b3

    def init_state(self, batch_size):
        """Random initial grids: (batch, H, W) of ints."""
        return torch.randint(0, self.num_colors,
                             (batch_size, self.grid_size, self.grid_size),
                             device=self.device)

    def step(self, state, params):
        """One NCA step for a batch of grids.

        state: (B, H, W) int tensor
        Returns: (B, H, W) int tensor
        """
        w1, b1, w2, b2, w3, b3 = params
        B, H, W = state.shape
        C = self.num_colors

        # One-hot encode: (B, C, H, W) for conv2d
        state_oh = F.one_hot(state.long(), C).float()  # (B, H, W, C)
        state_oh = state_oh.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Conv 3x3 with wrap padding — each rule has its own kernel
        # Pad with wrap
        state_oh_pad = F.pad(state_oh, (1, 1, 1, 1), mode='circular')

        # Batched grouped conv: reshape so each rule is a separate group
        # Input: (1, B*C, H+2, W+2), Weight: (B*4, C, 3, 3), groups=B
        x = state_oh_pad.reshape(1, B * C, H + 2, W + 2)
        w = w1  # (B, 4, C, 3, 3)  — already in right shape for grouped conv
        w = w.reshape(B * 4, C, 3, 3)
        x = F.conv2d(x, w, groups=B)  # (1, B*4, H, W)
        x = x.reshape(B, 4, H, W)
        x = x + b1[:, :, None, None]

        # ReLU + 1x1 conv (as matmul)
        x = F.relu(x)
        # x: (B, 4, H, W) -> (B, H*W, 4) @ w2^T -> (B, H*W, 16)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, 4)
        x = torch.bmm(x, w2.transpose(1, 2)) + b2[:, None, :]
        x = F.relu(x)

        # 1x1 conv -> logits
        x = torch.bmm(x, w3.transpose(1, 2)) + b3[:, None, :]  # (B, H*W, C)
        logits = x.reshape(B, H, W, C)

        # Add identity bias (0.0) and apply temperature (1.0)
        logits = logits + state_oh.permute(0, 2, 3, 1) * 0.0  # identity_bias=0

        # Gumbel-max sampling
        u = torch.rand_like(logits).clamp(1e-10, 1.0)
        gumbel = -torch.log(-torch.log(u))
        new_state = (logits + gumbel).argmax(dim=-1)  # (B, H, W)

        return new_state

    @torch.no_grad()
    def rollout(self, params, num_steps, dt=2, start_step=0, batch_size=None):
        """Run simulation for a batch of rules, return snapshots.

        Returns: (B, num_steps, H, W) int tensor
        """
        B = params[0].shape[0] if batch_size is None else batch_size
        state = self.init_state(B)

        # Burn-in
        for _ in range(start_step):
            state = self.step(state, params)

        # Collect snapshots
        snapshots = []
        for i in range(num_steps * dt):
            if i % dt == 0:
                snapshots.append(state.clone())
            state = self.step(state, params)

        return torch.stack(snapshots, dim=1)  # (B, num_steps, H, W)


# ── Tokenizer (GPU) ──────────────────────────────────────────────────────────

class NCATokenizerGPU:
    """Batch tokenizer on GPU."""

    def __init__(self, patch=2, num_colors=10, device="cuda"):
        self.patch = patch
        self.num_colors = num_colors
        self.start_tk = num_colors ** (patch ** 2)      # 10000
        self.end_tk = num_colors ** (patch ** 2) + 1     # 10001
        self.vocab_size = self.end_tk + 1                # 10002
        self.device = device
        self._powers = (num_colors ** torch.arange(patch * patch,
                        device=device)).long()

    def encode_batch(self, snapshots):
        """Encode batched trajectories into token sequences.

        snapshots: (B, T, H, W) int tensor
        Returns: list of 1D numpy arrays (one per rule)
        """
        B, T, H, W = snapshots.shape
        pH, pW = H // self.patch, W // self.patch
        ppg = pH * pW  # patches per grid
        tpg = ppg + 2  # tokens per grid (with start/end)

        # Reshape into patches: (B, T, pH, patch, pW, patch)
        g = snapshots.reshape(B, T, pH, self.patch, pW, self.patch)
        g = g.permute(0, 1, 2, 4, 3, 5).reshape(B, T, ppg, self.patch ** 2)

        # Compute patch tokens
        patch_tokens = (g.long() * self._powers).sum(dim=-1)  # (B, T, ppg)

        # Build full sequences with start/end tokens
        # (B, T, tpg) where each grid is [start, ...patches..., end]
        out = torch.empty(B, T, tpg, dtype=torch.long, device=self.device)
        out[:, :, 0] = self.start_tk
        out[:, :, 1:1+ppg] = patch_tokens
        out[:, :, 1+ppg] = self.end_tk

        # Flatten each rule's trajectory
        out_flat = out.reshape(B, T * tpg)  # (B, total_tokens_per_rule)
        return out_flat


# ── Complexity filter (batch, on CPU for np compatibility) ────────────────────

def batch_entropy_filter(token_batch, lower=5.0, upper=9.0):
    """Fast entropy-based filter. Returns boolean mask.

    token_batch: (B, L) numpy int array
    Calibrated against gzip [0.4, 0.9] — correlation 0.987.
    """
    B, L = token_batch.shape
    mask = np.ones(B, dtype=bool)
    for i in range(B):
        _, counts = np.unique(token_batch[i], return_counts=True)
        probs = counts / counts.sum()
        entropy = -(probs * np.log2(probs)).sum()
        if entropy < lower or entropy > upper:
            mask[i] = False
    return mask


def _compute_entropy_gpu(token_batch, vocab_size=10002):
    """Compute per-sequence entropy on GPU. Returns (B,) float tensor."""
    B, L = token_batch.shape
    counts = torch.zeros(B, vocab_size, device=token_batch.device)
    counts.scatter_add_(1, token_batch, torch.ones_like(token_batch, dtype=torch.float))
    probs = counts / L
    log_probs = torch.where(probs > 0, torch.log2(probs), torch.zeros_like(probs))
    return -(probs * log_probs).sum(dim=1)  # (B,)


def batch_entropy_filter_gpu(token_batch, lower=5.0, upper=9.0, vocab_size=10002):
    """Fully GPU entropy filter. token_batch: (B, L) long tensor on device."""
    entropy = _compute_entropy_gpu(token_batch, vocab_size)
    return (entropy >= lower) & (entropy <= upper)


# ── Main generation ──────────────────────────────────────────────────────────

def generate_nca_data(num_tokens, grid_size=12, num_colors=10, patch=2,
                      dt=2, steps_per_rule=10, start_step=0,
                      complexity_lower=0.4, complexity_upper=0.9,
                      seed=42, seq_len=1025, val_fraction=0.05,
                      device="cuda", batch_size=512,
                      filter_mode="gzip", **kwargs):
    """Generate NCA data using GPU-accelerated simulation.

    filter_mode: "gzip" (original, slower but exact), "entropy" (fast approx)
    """
    torch.manual_seed(seed)

    sim = NCASimulatorGPU(grid_size=grid_size, num_colors=num_colors, device=device)
    tok = NCATokenizerGPU(patch=patch, num_colors=num_colors, device=device)

    tokens_per_grid = 1 + (grid_size // patch) ** 2 + 1  # 38
    tokens_per_rule = tokens_per_grid * steps_per_rule    # 380

    print(f"Grid: {grid_size}x{grid_size}, colors: {num_colors}, patch: {patch}x{patch}")
    print(f"Vocab size: {tok.vocab_size} (0..{tok.end_tk})")
    print(f"Tokens per grid: {tokens_per_grid}, per rule: {tokens_per_rule}")
    print(f"Target: {num_tokens:,} tokens")
    print(f"Filter: {filter_mode} [{complexity_lower}, {complexity_upper}]")
    print(f"GPU batch size: {batch_size}")

    all_tokens = []
    all_entropies = []
    total_generated = 0
    total_accepted = 0
    total_rejected = 0
    t0 = time.time()

    while total_generated < num_tokens:
        # Generate batch of rules
        params = sim.random_params(batch_size)
        snapshots = sim.rollout(params, num_steps=steps_per_rule,
                                dt=dt, start_step=start_step)  # (B, T, H, W)

        # Tokenize on GPU
        token_seqs = tok.encode_batch(snapshots)  # (B, L)

        # Compute per-rule entropy on GPU (always, for sorting)
        rule_entropy = _compute_entropy_gpu(token_seqs, vocab_size=tok.vocab_size)

        # Filter
        if filter_mode == "entropy":
            mask = (rule_entropy >= 5.0) & (rule_entropy <= 8.0)
            accepted = token_seqs[mask].cpu().numpy().astype(np.int32)
            accepted_ent = rule_entropy[mask].cpu().numpy()
        else:
            # gzip filter — move to CPU for gzip compression
            token_np = token_seqs.cpu().numpy().astype(np.int16)
            mask_list = []
            for i in range(token_np.shape[0]):
                ratio = gzip_complexity(token_np[i].tobytes())
                mask_list.append(complexity_lower <= ratio <= complexity_upper)
            mask_t = torch.tensor(mask_list)
            accepted = token_seqs[mask_t].cpu().numpy().astype(np.int32)
            accepted_ent = rule_entropy[mask_t].cpu().numpy()
        n_accepted = accepted.shape[0]
        n_rejected = batch_size - n_accepted

        if n_accepted > 0:
            all_tokens.append(accepted.reshape(-1))
            all_entropies.extend(
                np.repeat(accepted_ent, tokens_per_rule).tolist()
                if n_accepted > 0 else []
            )
            total_generated += accepted.size

        total_accepted += n_accepted
        total_rejected += n_rejected

        elapsed = time.time() - t0
        rate = total_generated / elapsed if elapsed > 0 else 0
        accept_rate = total_accepted / max(1, total_accepted + total_rejected)

        if total_accepted % (batch_size * 10) < batch_size or total_generated >= num_tokens:
            print(f"  {total_generated:>12,} / {num_tokens:,} tokens | "
                  f"accept {accept_rate:.1%} | {rate:,.0f} tok/s | "
                  f"{elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"\nDone: {total_generated:,} tokens in {elapsed:.1f}s")
    print(f"Acceptance: {total_accepted}/{total_accepted+total_rejected} "
          f"({total_accepted/max(1,total_accepted+total_rejected):.1%})")

    # Pack into sequences
    flat = np.concatenate(all_tokens)
    num_seqs = len(flat) // seq_len
    sequences = flat[:num_seqs * seq_len].reshape(num_seqs, seq_len)
    print(f"Packed into {num_seqs:,} sequences of length {seq_len}")

    # Compute per-sequence entropy (average of constituent rules)
    ent_flat = np.array(all_entropies[:len(flat)])
    seq_entropies = ent_flat[:num_seqs * seq_len].reshape(num_seqs, seq_len).mean(axis=1)

    # Sort by entropy (low → high) so callers can slice for curriculum
    sort_idx = np.argsort(seq_entropies)
    sequences = sequences[sort_idx]
    seq_entropies = seq_entropies[sort_idx]
    print(f"Sorted by entropy: [{seq_entropies[0]:.2f}, {seq_entropies[-1]:.2f}]")

    # Split train/val (random split, preserving sort within each)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(num_seqs * val_fraction))
    n_train = num_seqs - n_val
    indices = rng.permutation(num_seqs)
    train_idx = np.sort(indices[:n_train])  # sort to preserve entropy order
    val_idx = np.sort(indices[n_train:])

    train_tokens = torch.tensor(sequences[train_idx], dtype=torch.int32)
    val_tokens = torch.tensor(sequences[val_idx], dtype=torch.int32)
    train_ent = torch.tensor(seq_entropies[train_idx], dtype=torch.float32)
    val_ent = torch.tensor(seq_entropies[val_idx], dtype=torch.float32)
    print(f"Train: {train_tokens.shape}, Val: {val_tokens.shape}")

    return train_tokens, val_tokens, train_ent, val_ent


def main():
    parser = argparse.ArgumentParser(description="GPU NCA data generation")
    parser.add_argument("--output-dir", type=str, default="./nca_data")
    parser.add_argument("--num-tokens", type=int, default=164_000_000)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--num-colors", type=int, default=10)
    parser.add_argument("--patch", type=int, default=2)
    parser.add_argument("--dt", type=int, default=2)
    parser.add_argument("--steps-per-rule", type=int, default=10)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=1025)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_tokens, val_tokens, train_ent, val_ent = generate_nca_data(
        num_tokens=args.num_tokens,
        grid_size=args.grid_size,
        num_colors=args.num_colors,
        patch=args.patch,
        dt=args.dt,
        steps_per_rule=args.steps_per_rule,
        start_step=args.start_step,
        seed=args.seed,
        seq_len=args.seq_len,
        val_fraction=args.val_fraction,
        device=args.device,
        batch_size=args.batch_size,
    )

    train_path = os.path.join(args.output_dir, "nca_train.pt")
    val_path = os.path.join(args.output_dir, "nca_val.pt")
    torch.save({"tokens": train_tokens, "entropy": train_ent}, train_path)
    torch.save({"tokens": val_tokens, "entropy": val_ent}, val_path)
    print(f"\nSaved: {train_path} ({os.path.getsize(train_path) / 1e6:.1f} MB)")
    print(f"Saved: {val_path} ({os.path.getsize(val_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
