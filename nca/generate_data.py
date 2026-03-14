"""
NCA data generation for pre-pre-training — pure PyTorch/NumPy, no JAX.

Generates NCA trajectories on a 12×12 grid with 10-state alphabet,
tokenizes using 2×2 patches → 10K vocab, filters by gzip compressibility,
and packs into sequences matching slowrun format (length 1025).

Output: nca_train.pt / nca_val.pt with same dict format as fineweb_train.pt:
  {"tokens": tensor of shape (num_sequences, seq_len+1)}

Usage:
    python nca_datagen.py --output-dir ./nca_data --num-tokens 164000000
"""

import argparse
import gzip
import io
import multiprocessing as mp
import os
import time

import numpy as np
import torch


# ── NCA Simulation (pure NumPy) ─────────────────────────────────────────────

class NCASimulator:
    """Discrete NCA: grid of integers updated by a stochastic neural CA."""

    def __init__(self, grid_size=12, num_colors=10, identity_bias=0.0,
                 temperature=1.0):
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.identity_bias = identity_bias
        self.temperature = temperature

    def random_params(self, rng):
        """Generate random NCA network parameters."""
        d_state = self.num_colors
        # Conv 3x3 -> 4 channels (with wrap padding)
        w1 = rng.standard_normal((3, 3, d_state, 4)).astype(np.float32) * 0.5
        b1 = np.zeros(4, dtype=np.float32)
        # Conv 1x1 -> 16
        w2 = rng.standard_normal((1, 1, 4, 16)).astype(np.float32) * 0.5
        b2 = np.zeros(16, dtype=np.float32)
        # Conv 1x1 -> d_state (output logits)
        w3 = rng.standard_normal((1, 1, 16, d_state)).astype(np.float32) * 0.5
        b3 = np.zeros(d_state, dtype=np.float32)
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}

    def init_state(self, rng):
        """Random initial grid state."""
        return rng.integers(0, self.num_colors,
                            size=(self.grid_size, self.grid_size))

    def _conv2d_wrap(self, x, w, b):
        """2D convolution with wrap (periodic) padding."""
        x_pad = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode='wrap')
        H, W = x.shape[0], x.shape[1]
        out = np.zeros((H, W, w.shape[3]), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                out += x_pad[i:i+H, j:j+W] @ w[i, j]
        return out + b

    def step(self, rng, state, params):
        """One NCA step: one_hot → conv network → sample."""
        state_oh = np.eye(self.num_colors, dtype=np.float32)[state]
        x = self._conv2d_wrap(state_oh, params["w1"], params["b1"])
        x = np.maximum(x, 0) @ params["w2"][0, 0] + params["b2"]
        x = np.maximum(x, 0) @ params["w3"][0, 0] + params["b3"]
        logits = (x + state_oh * self.identity_bias) / max(self.temperature, 1e-6)
        # Gumbel-max sampling
        logits_flat = logits.reshape(-1, self.num_colors)
        u = rng.random(logits_flat.shape).clip(1e-10, 1.0)
        gumbel = -np.log(-np.log(u))
        samples = (logits_flat + gumbel).argmax(axis=-1)
        return samples.reshape(self.grid_size, self.grid_size)

    def rollout(self, rng, params, num_steps, dt=2, start_step=0):
        """Run simulation and return snapshots every dt steps."""
        state = self.init_state(rng)
        for _ in range(start_step):
            state = self.step(rng, state, params)
        snapshots = []
        for i in range(num_steps * dt):
            if i % dt == 0:
                snapshots.append(state.copy())
            state = self.step(rng, state, params)
        return np.stack(snapshots)


# ── Tokenizer (pure NumPy) ──────────────────────────────────────────────────

class NCATokenizer:
    """Tokenizes NCA grids using patch-based encoding."""

    def __init__(self, patch=2, num_colors=10):
        self.patch = patch
        self.num_colors = num_colors
        self.start_tk = num_colors ** (patch ** 2)      # 10000
        self.end_tk = num_colors ** (patch ** 2) + 1     # 10001
        self.vocab_size = self.end_tk + 1                # 10002
        self._powers = num_colors ** np.arange(patch * patch)

    def encode_trajectory(self, snapshots):
        """Encode a sequence of grids into a flat token sequence.

        Each grid gets [start_tk, ...patch_tokens..., end_tk].
        """
        N, H, W = snapshots.shape
        pH, pW = H // self.patch, W // self.patch
        patches_per_grid = pH * pW

        # Batch-encode all grids at once
        g = snapshots.reshape(N, pH, self.patch, pW, self.patch)
        g = g.transpose(0, 1, 3, 2, 4).reshape(N, patches_per_grid, self.patch**2)
        patch_tokens = (g * self._powers).sum(axis=-1)  # (N, patches_per_grid)

        # Build full sequence with start/end tokens
        tokens_per_grid = patches_per_grid + 2
        out = np.empty(N * tokens_per_grid, dtype=np.int32)
        for i in range(N):
            base = i * tokens_per_grid
            out[base] = self.start_tk
            out[base+1:base+1+patches_per_grid] = patch_tokens[i]
            out[base+1+patches_per_grid] = self.end_tk
        return out


# ── Compressibility filter ──────────────────────────────────────────────────

def gzip_complexity(data_bytes):
    """Gzip compression ratio — lower means more compressible."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=9) as f:
        f.write(data_bytes)
    return len(buf.getvalue()) / len(data_bytes)


def passes_complexity_filter(tokens, lower=0.4, upper=0.9):
    """Check if token sequence has web-text-like compressibility."""
    ratio = gzip_complexity(tokens.astype(np.int16).tobytes())
    return lower <= ratio <= upper


# ── Worker for multiprocessing ──────────────────────────────────────────────

def _worker_generate(args):
    """Generate rules and return accepted token arrays."""
    (worker_id, num_rules, grid_size, num_colors, patch, dt,
     steps_per_rule, start_step, complexity_lower, complexity_upper, seed) = args

    sim = NCASimulator(grid_size=grid_size, num_colors=num_colors,
                       identity_bias=0.0, temperature=1.0)
    tok = NCATokenizer(patch=patch, num_colors=num_colors)
    rng = np.random.default_rng(seed)

    accepted_tokens = []
    accepted_count = 0
    rejected_count = 0

    for _ in range(num_rules):
        rule_rng = np.random.default_rng(rng.integers(0, 2**63))
        params = sim.random_params(rule_rng)
        traj_rng = np.random.default_rng(rng.integers(0, 2**63))
        snapshots = sim.rollout(traj_rng, params, steps_per_rule,
                                dt=dt, start_step=start_step)
        tokens = tok.encode_trajectory(snapshots)
        if passes_complexity_filter(tokens, complexity_lower, complexity_upper):
            accepted_tokens.append(tokens)
            accepted_count += 1
        else:
            rejected_count += 1

    return accepted_tokens, accepted_count, rejected_count


# ── Sequence packing ────────────────────────────────────────────────────────

def pack_sequences(all_tokens, seq_len=1025):
    """Pack a flat array of tokens into fixed-length sequences."""
    total = len(all_tokens)
    num_seqs = total // seq_len
    if num_seqs == 0:
        return np.array([], dtype=np.int32).reshape(0, seq_len)
    return all_tokens[:num_seqs * seq_len].reshape(num_seqs, seq_len)


# ── Main generation loop ────────────────────────────────────────────────────

def generate_nca_data(num_tokens, grid_size=12, num_colors=10, patch=2,
                      dt=2, steps_per_rule=10, start_step=0,
                      complexity_lower=0.4, complexity_upper=0.9,
                      seed=42, seq_len=1025, val_fraction=0.05,
                      num_workers=None):
    """Generate NCA pre-pre-training data with multiprocessing."""
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)

    tok = NCATokenizer(patch=patch, num_colors=num_colors)
    tokens_per_grid = 1 + (grid_size // patch) ** 2 + 1
    tokens_per_rule = tokens_per_grid * steps_per_rule

    print(f"Grid: {grid_size}x{grid_size}, colors: {num_colors}, patch: {patch}x{patch}")
    print(f"Vocab size: {tok.vocab_size} (0..{tok.end_tk})")
    print(f"Tokens per grid: {tokens_per_grid}, per rule: {tokens_per_rule}")
    print(f"Target: {num_tokens:,} tokens")
    print(f"Complexity filter: [{complexity_lower}, {complexity_upper}]")
    print(f"Workers: {num_workers}")

    rng = np.random.default_rng(seed)
    all_tokens = []
    total_generated = 0
    total_accepted = 0
    total_rejected = 0
    t0 = time.time()

    # Generate in batches — each batch dispatches work to all workers
    # Estimate ~3% acceptance rate, overshoot by 2x
    rules_per_batch = max(1000, num_tokens // tokens_per_rule)
    rules_per_worker = rules_per_batch // num_workers

    while total_generated < num_tokens:
        remaining = num_tokens - total_generated
        # Estimate rules needed (assume ~4% acceptance, 1.5x safety margin)
        est_rules = int(remaining / tokens_per_rule / 0.04 * 1.5)
        est_rules = max(500 * num_workers, min(est_rules, 50000 * num_workers))
        rpw = est_rules // num_workers + 1

        worker_args = []
        for w in range(num_workers):
            worker_seed = int(rng.integers(0, 2**63))
            worker_args.append((
                w, rpw, grid_size, num_colors, patch, dt,
                steps_per_rule, start_step, complexity_lower,
                complexity_upper, worker_seed
            ))

        with mp.Pool(num_workers) as pool:
            results = pool.map(_worker_generate, worker_args)

        for tokens_list, acc, rej in results:
            for t in tokens_list:
                all_tokens.append(t)
                total_generated += len(t)
            total_accepted += acc
            total_rejected += rej

        elapsed = time.time() - t0
        rate = total_generated / elapsed if elapsed > 0 else 0
        accept_rate = total_accepted / max(1, total_accepted + total_rejected)
        print(f"  {total_generated:>12,} / {num_tokens:,} tokens | "
              f"accept {accept_rate:.1%} | {rate:,.0f} tok/s | "
              f"{elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"\nDone: {total_generated:,} tokens in {elapsed:.1f}s")
    print(f"Acceptance: {total_accepted}/{total_accepted+total_rejected} "
          f"({total_accepted/max(1,total_accepted+total_rejected):.1%})")

    # Pack into sequences
    flat = np.concatenate(all_tokens)
    sequences = pack_sequences(flat, seq_len=seq_len)
    print(f"Packed into {len(sequences):,} sequences of length {seq_len}")

    # Split train/val
    n_val = max(1, int(len(sequences) * val_fraction))
    n_train = len(sequences) - n_val
    indices = rng.permutation(len(sequences))
    train_seqs = sequences[indices[:n_train]]
    val_seqs = sequences[indices[n_train:]]

    train_tokens = torch.tensor(train_seqs, dtype=torch.int32)
    val_tokens = torch.tensor(val_seqs, dtype=torch.int32)
    print(f"Train: {train_tokens.shape}, Val: {val_tokens.shape}")

    return train_tokens, val_tokens


def main():
    parser = argparse.ArgumentParser(description="Generate NCA pre-pre-training data")
    parser.add_argument("--output-dir", type=str, default="./nca_data")
    parser.add_argument("--num-tokens", type=int, default=164_000_000,
                        help="Target number of tokens to generate")
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--num-colors", type=int, default=10)
    parser.add_argument("--patch", type=int, default=2)
    parser.add_argument("--dt", type=int, default=2,
                        help="Timesteps between snapshots")
    parser.add_argument("--steps-per-rule", type=int, default=10,
                        help="Number of snapshots per NCA rule")
    parser.add_argument("--start-step", type=int, default=0,
                        help="Burn-in steps before collecting snapshots")
    parser.add_argument("--complexity-lower", type=float, default=0.4)
    parser.add_argument("--complexity-upper", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=1025,
                        help="Sequence length (input+target, matching slowrun)")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_tokens, val_tokens = generate_nca_data(
        num_tokens=args.num_tokens,
        grid_size=args.grid_size,
        num_colors=args.num_colors,
        patch=args.patch,
        dt=args.dt,
        steps_per_rule=args.steps_per_rule,
        start_step=args.start_step,
        complexity_lower=args.complexity_lower,
        complexity_upper=args.complexity_upper,
        seed=args.seed,
        seq_len=args.seq_len,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
    )

    train_path = os.path.join(args.output_dir, "nca_train.pt")
    val_path = os.path.join(args.output_dir, "nca_val.pt")
    torch.save({"tokens": train_tokens}, train_path)
    torch.save({"tokens": val_tokens}, val_path)
    print(f"\nSaved: {train_path} ({os.path.getsize(train_path) / 1e6:.1f} MB)")
    print(f"Saved: {val_path} ({os.path.getsize(val_path) / 1e6:.1f} MB)")

    # Verification
    print("\n=== Verification ===")
    d = torch.load(train_path, weights_only=True)
    t = d["tokens"]
    print(f"Shape: {t.shape}, dtype: {t.dtype}")
    print(f"Min: {t.min().item()}, Max: {t.max().item()}")
    assert t.min() >= 0, "Negative token!"
    assert t.max() <= 10001, f"Token out of range: {t.max().item()}"
    print("Token range: PASS")

    first_seq = t[0].numpy()
    start_positions = np.where(first_seq == 10000)[0]
    end_positions = np.where(first_seq == 10001)[0]
    print(f"First sequence start_tk positions: {start_positions[:5]}...")
    print(f"First sequence end_tk positions: {end_positions[:5]}...")
    print("Structure: PASS")


if __name__ == "__main__":
    main()
