"""
Transfer trunk weights from NCA-pretrained checkpoint → fresh slowrun model.

Copies: transformer.h.*, ve_projs.*, resid_lambdas, x0_lambdas, skip_weights
Re-initializes: transformer.wte (normal 0,1), lm_head (normal 0,0.001)
Recomputes: RoPE buffers for target seq_len

Usage:
    python transfer_weights.py \
        --nca-checkpoint ./nca_checkpoints/nca_best.pt \
        --output ./nca_checkpoints/transferred.pt \
        --target-vocab-size 50257
"""

import argparse
import os

import torch

from model import GPT, GPTConfig


def transfer_weights(nca_ckpt_path, output_path, target_vocab_size=50257,
                     target_seq_len=2048):
    """Transfer trunk weights from NCA checkpoint to fresh GPT model."""
    # Load NCA checkpoint
    ckpt = torch.load(nca_ckpt_path, weights_only=False, map_location="cpu")
    nca_config = ckpt["config"]
    nca_state = ckpt["model_state_dict"]
    # Strip _orig_mod. prefix from torch.compile'd checkpoints
    nca_state = {k.removeprefix("_orig_mod."): v for k, v in nca_state.items()}
    print(f"NCA checkpoint: step={ckpt.get('step')}, "
          f"val_loss={ckpt.get('val_loss', '?')}")
    print(f"NCA config: vocab={nca_config.vocab_size}, "
          f"layers={nca_config.n_layer}, embd={nca_config.n_embd}")

    # Create target model with same architecture but different vocab
    target_config = GPTConfig(
        vocab_size=target_vocab_size,
        n_layer=nca_config.n_layer,
        n_head=nca_config.n_head,
        n_kv_head=nca_config.n_kv_head,
        n_embd=nca_config.n_embd,
        sequence_len=target_seq_len,
        dropout=nca_config.dropout,
    )
    target_model = GPT(target_config)
    target_model.init_weights()
    print(f"\nTarget config: vocab={target_vocab_size}, "
          f"seq_len={target_seq_len}")

    # Determine which keys to transfer vs re-initialize
    skip_prefixes = ("transformer.wte.", "lm_head.", "cos", "sin")
    transferred = []
    skipped = []

    target_state = target_model.state_dict()
    for key in nca_state:
        if any(key.startswith(p) for p in skip_prefixes):
            skipped.append(key)
            continue
        if key in target_state:
            if nca_state[key].shape == target_state[key].shape:
                target_state[key] = nca_state[key]
                transferred.append(key)
            else:
                print(f"  SHAPE MISMATCH: {key} "
                      f"nca={nca_state[key].shape} vs "
                      f"target={target_state[key].shape}")
                skipped.append(key)
        else:
            print(f"  NOT IN TARGET: {key}")
            skipped.append(key)

    target_model.load_state_dict(target_state)

    print(f"\nTransferred {len(transferred)} weight tensors:")
    for k in transferred:
        print(f"  {k}: {nca_state[k].shape}")
    print(f"\nSkipped {len(skipped)} (re-initialized):")
    for k in skipped:
        print(f"  {k}")

    # Verify trunk weights match
    print("\n=== Verification ===")
    target_state_check = target_model.state_dict()
    mismatches = 0
    for key in transferred:
        if not torch.equal(target_state_check[key], nca_state[key]):
            print(f"  MISMATCH: {key}")
            mismatches += 1
    assert mismatches == 0, f"{mismatches} weight mismatches!"
    print("Trunk weights match: PASS")

    # Verify embeddings are fresh (not from NCA)
    wte = target_state_check["transformer.wte.weight"]
    assert wte.shape[0] != nca_state["transformer.wte.weight"].shape[0], \
        "Embedding size should differ!"
    print("Embeddings re-initialized: PASS")

    # Verify forward pass works
    x = torch.randint(0, target_vocab_size, (2, 64))
    t = torch.randint(0, target_vocab_size, (2, 64))
    with torch.no_grad():
        loss = target_model(x, t)
    assert loss.isfinite(), f"Forward pass produced non-finite loss: {loss}"
    print(f"Forward pass loss: {loss.item():.4f}: PASS")

    # Save
    torch.save({
        "model_state_dict": target_model.state_dict(),
        "config": target_config,
        "source_nca_config": nca_config,
        "source_step": ckpt.get("step"),
        "source_val_loss": ckpt.get("val_loss"),
        "transferred_keys": transferred,
        "skipped_keys": skipped,
    }, output_path)
    print(f"\nSaved: {output_path} "
          f"({os.path.getsize(output_path) / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Transfer NCA weights to slowrun model")
    parser.add_argument("--nca-checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--target-vocab-size", type=int, default=50257)
    parser.add_argument("--target-seq-len", type=int, default=2048)
    args = parser.parse_args()

    transfer_weights(args.nca_checkpoint, args.output,
                     args.target_vocab_size, args.target_seq_len)


if __name__ == "__main__":
    main()
