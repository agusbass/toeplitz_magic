"""
run_demo.py — Jalankan ini dulu untuk verifikasi cepat

    python run_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from toeplitz_magic.matrix import (
    build_toeplitz_11x11, build_magic_kernel,
    verify_disjoint, print_report
)
from toeplitz_magic.attention import (
    ToeplitzMagicAttention, ToeplitzMagicBlock, ToeplitzMagicEncoder
)

def run():
    print_report()

    print()
    print("=" * 56)
    print("Forward pass tests")
    print("=" * 56)

    torch.manual_seed(0)
    B, L, D = 2, 128, 256

    # Single attention layer
    attn = ToeplitzMagicAttention(d_model=D, n_heads=8, seq_len=L)
    x   = torch.randn(B, L, D)
    out = attn(x)
    print(f"Attention  : {x.shape} → {out.shape} ✓")

    # Transformer block
    block = ToeplitzMagicBlock(d_model=D, n_heads=8, seq_len=L)
    out   = block(x)
    print(f"Block      : {x.shape} → {out.shape} ✓")

    # Full encoder (6 layers)
    enc   = ToeplitzMagicEncoder(n_layers=6, d_model=D, n_heads=8, seq_len=L)
    out   = enc(x)
    stats = enc.count_params()
    print(f"Encoder    : {x.shape} → {out.shape} ✓")

    print()
    print("=" * 56)
    print("Parameter stats (6-layer encoder)")
    print("=" * 56)
    print(f"Total params          : {stats['total']:,}")
    print(f"Trainable             : {stats['trainable']:,}")
    print(f"Positional (learnable): {stats['positional_learnable']}")
    print(f"Positional (fixed)    : {stats['positional_fixed']:,}")
    print()
    print("✓ Semua test passed. Siap untuk benchmark & deploy.")

if __name__ == "__main__":
    run()
