"""
attention.py — ToeplitzMagicAttention & ToeplitzMagicBlock

Drop-in replacement untuk RoPE / ALiBi / Sinusoidal.
- Zero learnable positional params (bias dari matriks deterministik)
- O(n log n) via Toeplitz structure
- Integer-friendly — semua bias dari integer matrix
- Bidirectional by default (simetris A[i][j]=A[j][i])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .matrix import build_toeplitz_bias, build_dual_domain_encoding


class ToeplitzMagicAttention(nn.Module):
    """
    Self-attention dengan Toeplitz positional bias deterministik.

    Penggunaan:
        layer = ToeplitzMagicAttention(d_model=256, n_heads=8, seq_len=512)
        out = layer(x)   # x: (B, L, d_model)

    Perbedaan vs standard attention:
        - Tidak ada rope/alibi params
        - positional bias di-cache sebagai buffer (tidak ditraining)
        - Semua nilai bias berasal dari integer matrix
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        seq_len: int = 512,
        dropout: float = 0.0,
        bias_scale: float = 0.1,
    ):
        """
        Args:
            d_model    : dimensi model (harus habis dibagi n_heads)
            n_heads    : jumlah attention heads
            seq_len    : panjang sekuens maksimum
            dropout    : dropout pada attention weights
            bias_scale : skala Toeplitz bias relatif terhadap attention scores
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model harus habis dibagi n_heads"

        self.d_model    = d_model
        self.n_heads    = n_heads
        self.d_head     = d_model // n_heads
        self.seq_len    = seq_len
        self.bias_scale = bias_scale

        # QKV + output projection — satu-satunya learnable params
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        # Toeplitz positional bias — FIXED, tidak ditraining
        self.register_buffer(
            "pos_bias",
            build_toeplitz_bias(seq_len)   # (seq_len, seq_len)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x    : (B, L, d_model)
            mask : (B, L) bool — True = posisi yang di-mask (padding)

        Returns:
            (B, L, d_model)
        """
        B, L, _ = x.shape
        assert L <= self.seq_len, (
            f"seq_len {L} melebihi maksimum {self.seq_len}. "
            "Buat ulang dengan seq_len lebih besar."
        )

        # Project ke Q, K, V
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # Reshape ke multi-head: (B, H, L, d_head)
        def reshape(t):
            return t.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention scores
        scale  = self.d_head ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, L, L)

        # ── Toeplitz positional bias ─────────────────────────────────
        # Deterministik, simetris, integer-derived
        bias = self.pos_bias[:L, :L] * self.bias_scale          # (L, L)
        scores = scores + bias.unsqueeze(0).unsqueeze(0)        # broadcast ke (B,H,L,L)

        # Padding mask
        if mask is not None:
            scores = scores.masked_fill(
                mask[:, None, None, :], float("-inf")
            )

        attn  = self.drop(F.softmax(scores, dim=-1))
        out   = torch.matmul(attn, v)                           # (B, H, L, d_head)
        out   = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out(out)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"d_head={self.d_head}, seq_len={self.seq_len}, "
            f"positional_params=0 (deterministik)"
        )


class ToeplitzMagicBlock(nn.Module):
    """
    Transformer block lengkap dengan ToeplitzMagicAttention.

    Pre-LN architecture (stabilitas training lebih baik).

    Penggunaan:
        block = ToeplitzMagicBlock(d_model=256, n_heads=8)
        out   = block(x)   # (B, L, d_model) → (B, L, d_model)
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        ff_dim:   int   = None,
        seq_len:  int   = 512,
        dropout:  float = 0.1,
        bias_scale: float = 0.1,
    ):
        super().__init__()
        ff_dim = ff_dim or d_model * 4

        self.attn = ToeplitzMagicAttention(
            d_model, n_heads, seq_len, dropout, bias_scale
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-LN + residual
        x = x + self.drop(self.attn(self.ln1(x), mask))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class ToeplitzMagicEncoder(nn.Module):
    """
    Stack N ToeplitzMagicBlock — siap sebagai encoder.

    Penggunaan:
        enc = ToeplitzMagicEncoder(
            n_layers=6, d_model=256, n_heads=8, seq_len=512
        )
        out = enc(x)   # (B, L, d_model)
    """

    def __init__(
        self,
        n_layers: int,
        d_model:  int,
        n_heads:  int,
        ff_dim:   int   = None,
        seq_len:  int   = 512,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            ToeplitzMagicBlock(d_model, n_heads, ff_dim, seq_len, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    def count_params(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pos_fixed = sum(
            b.numel()
            for layer in self.layers
            for name, b in layer.attn.named_buffers()
            if "pos_bias" in name
        )
        return {
            "total":          total,
            "trainable":      trainable,
            "positional_fixed": pos_fixed,
            "positional_learnable": 0,
        }
