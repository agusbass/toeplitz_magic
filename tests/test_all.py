"""
tests/test_all.py — Test suite lengkap

Jalankan:
    pytest tests/ -v
"""

import math
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toeplitz_magic.matrix import (
    build_toeplitz_11x11,
    build_magic_kernel,
    build_toeplitz_bias,
    build_dual_domain_encoding,
    verify_disjoint,
    TOEPLITZ_SEED,
    GROUND_TRUTH_11x11,
)
from toeplitz_magic.attention import (
    ToeplitzMagicAttention,
    ToeplitzMagicBlock,
    ToeplitzMagicEncoder,
)


# ─── Matrix tests ─────────────────────────────────────────────────────

class TestToeplitz:

    def test_ground_truth_match(self):
        T = build_toeplitz_11x11()
        gt = torch.tensor(GROUND_TRUTH_11x11, dtype=torch.float32)
        assert torch.allclose(T, gt), "Tidak cocok dengan ground truth"

    def test_symmetric(self):
        T = build_toeplitz_11x11()
        assert torch.allclose(T, T.T), "Matriks tidak simetris"

    def test_shape(self):
        T = build_toeplitz_11x11()
        assert T.shape == (11, 11)

    def test_unique_values(self):
        T = build_toeplitz_11x11()
        vals = sorted(set(T.flatten().int().tolist()))
        assert vals == [0, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    def test_hankel_property(self):
        """Setiap anti-diagonal harus bernilai konstan (Hankel: val[i][j]=f(i+j))."""
        T = build_toeplitz_11x11()
        for s in range(21):  # i+j bisa 0..20
            vals = [T[i, s-i].item() for i in range(11) if 0 <= s-i < 11]
            if vals:
                assert len(set(vals)) == 1, f"Anti-diagonal s={s} tidak konstan: {vals}"

    def test_seed_consistency(self):
        """val[i][j] harus = f((i+j) % 20) — formula Hankel."""
        from toeplitz_magic.matrix import _HANKEL_LUT
        T = build_toeplitz_11x11()
        for i in range(11):
            for j in range(11):
                expected = _HANKEL_LUT[(i + j) % 20]
                assert T[i, j].item() == expected, \
                    f"T[{i}][{j}]: expected {expected}, got {T[i,j].item()}"


class TestMagicKernel:

    def test_magic_constant(self):
        M = build_magic_kernel()
        for i in range(3):
            assert M[i].sum().item() == 15, f"Baris {i} bukan 15"
            assert M[:, i].sum().item() == 15, f"Kolom {i} bukan 15"
        assert M.diag().sum().item() == 15
        assert M.flip(1).diag().sum().item() == 15

    def test_determinant(self):
        M = build_magic_kernel()
        det = torch.linalg.det(M.double()).item()
        assert abs(det - (-360)) < 1e-6, f"Det = {det}"

    def test_trace(self):
        M = build_magic_kernel()
        assert M.diag().sum().item() == 15

    def test_unique_values(self):
        M = build_magic_kernel()
        vals = sorted(set(M.flatten().int().tolist()))
        assert vals == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_shape(self):
        M = build_magic_kernel()
        assert M.shape == (3, 3)


class TestDomainProperties:

    def test_disjoint(self):
        assert verify_disjoint(), "Domain Toeplitz dan Magic tidak disjoint!"

    def test_union_coverage(self):
        T = build_toeplitz_11x11()
        M = build_magic_kernel()
        union = set(T.flatten().int().tolist()) | set(M.flatten().int().tolist())
        # Harus mencakup {0-9, 11-19}, missing hanya {10}
        expected = set(range(0, 10)) | set(range(11, 20))
        assert union == expected, f"Union tidak sesuai: {sorted(union)}"

    def test_missing_value_is_10(self):
        T = build_toeplitz_11x11()
        M = build_magic_kernel()
        union = set(T.flatten().int().tolist()) | set(M.flatten().int().tolist())
        full = set(range(0, 20))
        missing = full - union
        assert missing == {10}, f"Nilai yang hilang: {missing}"


class TestBiasAndEncoding:

    def test_toeplitz_bias_shape(self):
        for n in [64, 128, 512]:
            bias = build_toeplitz_bias(n)
            assert bias.shape == (n, n)

    def test_toeplitz_bias_symmetric(self):
        bias = build_toeplitz_bias(64)
        assert torch.allclose(bias, bias.T)

    def test_toeplitz_bias_range(self):
        bias = build_toeplitz_bias(64)
        assert bias.min() >= 0.0
        assert bias.max() <= 1.0

    def test_dual_domain_shape(self):
        enc = build_dual_domain_encoding(seq_len=32, d_model=64)
        assert enc.shape == (32, 64)

    def test_dual_domain_range(self):
        enc = build_dual_domain_encoding(seq_len=32, d_model=64)
        assert enc.min() >= 0.0
        assert enc.max() <= 1.0


# ─── Attention tests ──────────────────────────────────────────────────

class TestAttentionLayer:

    def setup_method(self):
        torch.manual_seed(42)
        self.B, self.L, self.D = 2, 64, 128
        self.model = ToeplitzMagicAttention(
            d_model=self.D, n_heads=4, seq_len=self.L
        )

    def test_forward_shape(self):
        x = torch.randn(self.B, self.L, self.D)
        out = self.model(x)
        assert out.shape == (self.B, self.L, self.D)

    def test_no_positional_learnable_params(self):
        """pos_bias harus buffer, bukan parameter."""
        param_names = [n for n, _ in self.model.named_parameters()]
        assert not any("pos_bias" in n for n in param_names)

    def test_pos_bias_is_buffer(self):
        buffer_names = [n for n, _ in self.model.named_buffers()]
        assert "pos_bias" in buffer_names

    def test_pos_bias_symmetric(self):
        bias = self.model.pos_bias
        assert torch.allclose(bias, bias.T)

    def test_with_padding_mask(self):
        x    = torch.randn(self.B, self.L, self.D)
        mask = torch.zeros(self.B, self.L, dtype=torch.bool)
        mask[:, -4:] = True   # mask 4 token terakhir
        out  = self.model(x, mask=mask)
        assert out.shape == (self.B, self.L, self.D)

    def test_deterministic_output(self):
        """Output harus sama untuk input yang sama (eval mode)."""
        x = torch.randn(self.B, self.L, self.D)
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(x)
            out2 = self.model(x)
        assert torch.allclose(out1, out2)

    def test_gradient_flows(self):
        x = torch.randn(self.B, self.L, self.D, requires_grad=True)
        out = self.model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestTransformerBlock:

    def setup_method(self):
        torch.manual_seed(42)
        self.B, self.L, self.D = 2, 64, 128
        self.block = ToeplitzMagicBlock(
            d_model=self.D, n_heads=4, seq_len=self.L
        )

    def test_forward_shape(self):
        x = torch.randn(self.B, self.L, self.D)
        out = self.block(x)
        assert out.shape == (self.B, self.L, self.D)

    def test_residual_connection(self):
        """Output tidak boleh sama persis dengan input (ada transformasi)."""
        x = torch.randn(self.B, self.L, self.D)
        self.block.eval()
        with torch.no_grad():
            out = self.block(x)
        assert not torch.allclose(x, out)


class TestEncoder:

    def setup_method(self):
        torch.manual_seed(42)
        self.B, self.L, self.D = 2, 64, 128
        self.enc = ToeplitzMagicEncoder(
            n_layers=3, d_model=self.D, n_heads=4, seq_len=self.L
        )

    def test_forward_shape(self):
        x = torch.randn(self.B, self.L, self.D)
        out = self.enc(x)
        assert out.shape == (self.B, self.L, self.D)

    def test_param_count(self):
        stats = self.enc.count_params()
        assert stats["positional_learnable"] == 0
        assert stats["trainable"] > 0
        assert stats["positional_fixed"] > 0
        print(f"\nParams: {stats}")


# ─── Benchmark sederhana ──────────────────────────────────────────────

class TestBenchmark:

    @pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
    def test_throughput(self, seq_len):
        """Model harus bisa forward pass tanpa error untuk berbagai seq_len."""
        model = ToeplitzMagicAttention(d_model=64, n_heads=4, seq_len=seq_len)
        model.eval()
        x = torch.randn(1, seq_len, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, seq_len, 64)

    def test_parameter_efficiency(self):
        """Bandingkan jumlah params vs standard attention."""
        D, H, L = 256, 8, 512

        # Toeplitz Magic (tanpa positional params)
        tm = ToeplitzMagicAttention(d_model=D, n_heads=H, seq_len=L)
        tm_params = sum(p.numel() for p in tm.parameters() if p.requires_grad)

        # Standard attention (dengan learned positional embedding)
        class StandardAttn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv = torch.nn.Linear(D, 3*D, bias=False)
                self.out = torch.nn.Linear(D, D, bias=False)
                self.pos = torch.nn.Embedding(L, D)  # learnable pos
        sa_params = sum(p.numel() for p in StandardAttn().parameters())

        saving = (sa_params - tm_params) / sa_params * 100
        print(f"\nToeplitz Magic : {tm_params:,} params")
        print(f"Standard       : {sa_params:,} params")
        print(f"Penghematan    : {saving:.1f}%")
        assert tm_params < sa_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
