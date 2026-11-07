# ToeplitzMagicAttention

A drop-in replacement for RoPE / ALiBi / sinusoidal positional encoding.

- **Zero learnable positional parameters** — the bias comes from a deterministic integer matrix
- **O(n log n)** complexity via Toeplitz structure
- **Integer-friendly** — all bias values derive from integer matrices (edge/mobile ready)
- **Dual-domain disjoint encoding**

## Install

```bash
pip install torch
python -m pytest tests/
```

## Quick start

```python
from toeplitz_magic.attention import ToeplitzMagicAttention

layer = ToeplitzMagicAttention(d_model=256, n_heads=8, seq_len=512)
out = layer(x)  # x: (B, L, d_model)
```

## How it differs from standard attention

- No RoPE / ALiBi parameters
- Positional bias is cached as a non-trainable buffer
- Symmetric (bidirectional by default): `A[i][j] == A[j][i]`

## License

MIT License — see [LICENSE](LICENSE). © 2026 PT Raya Laksana Teknologi (ralafs.com).

For production/enterprise deployment and commercial support, contact demo@ralafs.com.
