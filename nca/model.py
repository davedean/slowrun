"""
Portable slowrun GPT model — extracted from slowrun/train.py.

Replaces FA3 with F.scaled_dot_product_attention so it runs on any GPU.
Keeps ALL architectural features: U-Net skips, VE projections, XSA,
attention gating, SwiGLU, residual/x0 lambdas, dupe layers.

Parameterized vocab_size: 10002 for NCA, 50257 for GPT-2.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ── Helpers ──────────────────────────────────────────────────────────────────

def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Value Embedding on alternating layers, last layer always included."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


# ── SDPA attention (works on any GPU) ───────────────────────────────────────

def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """SDPA with sliding window support. q,k,v: (B, H, T, D)."""
    Tq, Tk = q.size(2), k.size(2)
    window = window_size[0]
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                              enable_gqa=enable_gqa)
    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                          enable_gqa=enable_gqa)


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """Drop-in replacement for flash_attn. q,k,v: (B, T, H, D)."""
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 50257
    n_layer: int = 30
    n_head: int = 14
    n_kv_head: int = 14
    n_embd: int = 1792
    window_pattern: str = "SSSL"
    dropout: float = 0.0


# ── Model ────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.ve_gate_channels = 32
        self.ve_gate = (nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
                        if has_ve(layer_idx, config.n_layer) else None)
        self.attn_gate_channels = 12
        self.attn_gate = nn.Linear(self.attn_gate_channels, self.n_head, bias=False)

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        # Value residual (ResFormer)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        y = flash_attn_func(q, k, v, causal=True, window_size=window_size)
        # XSA: remove self-value projection from attention output
        vn = F.normalize(v, dim=-1)
        if self.n_kv_head != self.n_head:
            vn = vn.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        y = y - (y * vn).sum(dim=-1, keepdim=True) * vn
        # Attention gate
        y = y * torch.sigmoid(
            self.attn_gate(x[..., :self.attn_gate_channels])
        ).unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 256 * ((8 * config.n_embd // 3 + 255) // 256)
        self.c_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.resid_dropout(self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab = ((config.vocab_size + pad_vocab_size_to - 1)
                        // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab != config.vocab_size:
            print(f"Padding vocab_size from {config.vocab_size} to {padded_vocab}")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.ve_projs = nn.ModuleDict({
            str(i): nn.Linear(config.n_embd, kv_dim, bias=False)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.encoder_layers = config.n_layer // 2
        self.skip_weights = nn.Parameter(torch.ones(self.encoder_layers))
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self._dupe_layers = None

    def set_dupe_layers(self, start, end):
        assert start >= self.encoder_layers, "dupe layers must be decoder-only"
        assert end <= self.config.n_layer
        self._dupe_layers = (start, end)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3**0.5 * self.config.n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_gate.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for proj in self.ve_projs.values():
            torch.nn.init.uniform_(proj.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            torch.nn.init.zeros_(block.attn.attn_gate.weight)
        self.skip_weights.fill_(1.0)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2,
                          dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = config.sequence_len, config.sequence_len // 2
        char_to_w = {"L": (long_w, 0), "S": (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]]
                 for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)
        return sizes

    def _run_decoder_layers(self, x, x0, cos_sin, encoder_outputs, start, end):
        """Run decoder layers [start, end), with U-Net skip connections."""
        for i in range(start, end):
            j = self.config.n_layer - 1 - i
            if 0 <= j < self.encoder_layers:
                x = x + self.skip_weights[i - self.encoder_layers] * encoder_outputs[j]
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.ve_projs[str(i)](x0) if str(i) in self.ve_projs else None
            x = self.transformer.h[i](x, ve, cos_sin, self.window_sizes[i])
        return x

    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx))
        x0 = x

        # Encoder half
        encoder_outputs = []
        for i in range(self.encoder_layers):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.ve_projs[str(i)](x0) if str(i) in self.ve_projs else None
            x = self.transformer.h[i](x, ve, cos_sin, self.window_sizes[i])
            encoder_outputs.append(x)

        # Decoder half
        dupe = self._dupe_layers
        if dupe is None:
            x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs,
                                         self.encoder_layers, self.config.n_layer)
        else:
            x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs,
                                         self.encoder_layers, dupe[1])
            x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs,
                                         dupe[0], dupe[1])
            x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs,
                                         dupe[0], dupe[1])
            x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs,
                                         dupe[1], self.config.n_layer)

        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)
        if targets is not None:
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-1,
                                   reduction=loss_reduction)
        return logits


# ── Verification ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Testing GPT model (portable, no FA3) ===\n")

    # Test 1: NCA vocab (10002)
    nca_config = GPTConfig(vocab_size=10002, n_layer=4, n_head=4,
                           n_kv_head=4, n_embd=256, sequence_len=1024)
    nca_model = GPT(nca_config)
    nca_model.init_weights()
    nca_params = sum(p.numel() for p in nca_model.parameters())
    print(f"NCA config: {nca_config}")
    print(f"NCA params: {nca_params:,}")

    x_nca = torch.randint(0, 10002, (2, 64))
    t_nca = torch.randint(0, 10002, (2, 64))
    loss_nca = nca_model(x_nca, t_nca)
    print(f"NCA forward pass loss: {loss_nca.item():.4f}")
    assert loss_nca.isfinite(), "NCA loss is not finite!"
    print("  PASS\n")

    # Test 2: GPT-2 vocab (50257)
    gpt_config = GPTConfig(vocab_size=50257, n_layer=4, n_head=4,
                           n_kv_head=4, n_embd=256, sequence_len=1024)
    gpt_model = GPT(gpt_config)
    gpt_model.init_weights()
    gpt_params = sum(p.numel() for p in gpt_model.parameters())
    print(f"GPT config: {gpt_config}")
    print(f"GPT params: {gpt_params:,}")

    x_gpt = torch.randint(0, 50257, (2, 64))
    t_gpt = torch.randint(0, 50257, (2, 64))
    loss_gpt = gpt_model(x_gpt, t_gpt)
    print(f"GPT forward pass loss: {loss_gpt.item():.4f}")
    assert loss_gpt.isfinite(), "GPT loss is not finite!"
    print("  PASS\n")

    # Test 3: Full-size config parameter count
    full_config = GPTConfig()  # n_layer=30, n_embd=1792, n_head=14
    full_model = GPT(full_config)
    full_params = sum(p.numel() for p in full_model.parameters())
    print(f"Full config params: {full_params:,}")
    # Expected ~1.4B for the slowrun architecture
    assert full_params > 1_000_000_000, f"Full model too small: {full_params:,}"
    assert full_params < 2_000_000_000, f"Full model too large: {full_params:,}"
    print("  PASS (param count in expected range)\n")

    # Test 4: Dupe layers
    dupe_config = GPTConfig(vocab_size=10002, n_layer=8, n_head=4,
                            n_kv_head=4, n_embd=256, sequence_len=1024)
    dupe_model = GPT(dupe_config)
    dupe_model.init_weights()
    dupe_model.set_dupe_layers(5, 7)
    x_d = torch.randint(0, 10002, (2, 64))
    t_d = torch.randint(0, 10002, (2, 64))
    loss_d = dupe_model(x_d, t_d)
    print(f"Dupe layers forward pass loss: {loss_d.item():.4f}")
    assert loss_d.isfinite(), "Dupe loss is not finite!"
    print("  PASS\n")

    print("All tests passed!")
