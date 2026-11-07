"""
run_tier_adapter.py — Demo TM-Compress Tier Adapter

    python run_tier_adapter.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from toeplitz_magic.tier_adapter import (
    TMAdapter, check_all_tiers,
    StreamingToeplitzBias, detect_hardware
)
from toeplitz_magic.attention import ToeplitzMagicEncoder


def main():
    # 1. Auto-detect & report
    adapter = TMAdapter.auto(verbose=True)

    # 2. Tampilkan semua tier vs semua model
    print("  ALL MODELS — COVERAGE TABLE")
    check_all_tiers()

    # 3. Demo streaming bias memory savings
    print("  STREAMING BIAS — MEMORY SAVINGS")
    print("─" * 52)
    bias = StreamingToeplitzBias()
    for n in [128, 512, 1024, 2048, 4096]:
        m = bias.memory_bytes(n)
        print(f"  n={n:<5} streaming={m['streaming_bytes']}B  "
              f"full={m['full_matrix_bytes']//1024}KB  "
              f"hemat={m['saving_pct']:.2f}%")

    # 4. Forward pass dengan streaming
    print("\n  FORWARD PASS TEST (streaming mode)")
    print("─" * 52)
    torch.manual_seed(0)
    B, L, D = 1, 128, 256
    enc  = ToeplitzMagicEncoder(n_layers=3, d_model=D, n_heads=8, seq_len=L)
    wrapped = adapter.wrap(enc)

    x   = torch.randn(B, L, D)
    out = wrapped(x)
    print(f"  Input  : {x.shape}")
    print(f"  Output : {out.shape} ✓")

    # 5. Fit check
    print("\n  FIT CHECK — apakah model muat di hardware Anda")
    print("─" * 52)
    result = adapter.fits_in_hardware(enc)
    status = "✓ MUAT" if result["fits"] else "✗ TIDAK MUAT"
    print(f"  Model params   : {result['params_B']:.2f}B")
    print(f"  VRAM BF16      : {result['vram_bf16_gb']:.2f} GB")
    print(f"  VRAM TM-Comp   : {result['vram_compressed_gb']:.2f} GB")
    print(f"  Hardware avail : {result['available_gb']:.2f} GB")
    print(f"  Status         : {status}")
    print(f"  Headroom       : {result['headroom_gb']:.2f} GB")
    print(f"  Compression    : {result['compression_ratio']:.1f}×")
    print()
    print("✓ TM-Compress Tier Adapter siap deploy.")


if __name__ == "__main__":
    main()
