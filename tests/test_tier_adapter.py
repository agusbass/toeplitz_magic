"""tests/test_tier_adapter.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from toeplitz_magic.tier_adapter import (
    detect_hardware, TMAdapter, StreamingToeplitzBias,
    estimate_vram_bf16, estimate_vram_compressed,
    params_to_tier, TIERS, check_all_tiers
)
from toeplitz_magic.attention import ToeplitzMagicEncoder


class TestHardwareDetect:
    def test_detect_runs(self):
        hw = detect_hardware()
        assert hw.ram_gb > 0
        assert hw.cpu_cores > 0
        assert hw.torch_version != ""

    def test_gpu_fields(self):
        hw = detect_hardware()
        if hw.has_gpu:
            assert hw.gpu_vram_total > 0
            assert hw.gpu_name != "none"


class TestTierSelection:
    @pytest.mark.parametrize("params_b,expected_tier", [
        (0.5e9, 1), (2e9, 1), (3e9, 1),
        (7e9, 2),  (13e9, 2),
        (30e9, 3), (70e9, 3),
        (100e9, 4),(200e9, 4),
        (405e9, 5),(500e9, 5),
    ])
    def test_tier_mapping(self, params_b, expected_tier):
        assert params_to_tier(int(params_b)) == expected_tier

    def test_force_tier(self):
        for i in range(1, 6):
            adapter = TMAdapter.force_tier(i)
            assert adapter.tier.tier_id == i


class TestStreamingBias:
    def test_shape(self):
        bias = StreamingToeplitzBias()
        for n in [16, 64, 128, 512]:
            b = bias(n)
            assert b.shape == (n, n)

    def test_symmetric(self):
        bias = StreamingToeplitzBias()
        b = bias(64)
        assert torch.allclose(b, b.T)

    def test_range(self):
        bias = StreamingToeplitzBias()
        b = bias(64)
        assert b.min() >= 0.0
        assert b.max() <= 1.0

    def test_memory_savings(self):
        bias = StreamingToeplitzBias()
        m = bias.memory_bytes(4096)
        assert m["saving_pct"] > 99.0    # harus hemat >99%
        assert m["streaming_bytes"] == 84  # 21 float32 (Hankel LUT) = 84 bytes

    def test_deterministic(self):
        bias = StreamingToeplitzBias()
        b1 = bias(64)
        b2 = bias(64)
        assert torch.allclose(b1, b2)


class TestAdapter:
    def test_auto_runs(self):
        adapter = TMAdapter.auto(verbose=False)
        assert adapter.tier.tier_id in range(1, 6)

    def test_report_string(self):
        adapter = TMAdapter.auto(verbose=False)
        r = adapter.report()
        assert "TM-Compress" in r
        assert "Tier" in r

    def test_wrap_encoder(self):
        adapter = TMAdapter.force_tier(1)
        enc = ToeplitzMagicEncoder(n_layers=2, d_model=64, n_heads=4, seq_len=32)
        wrapped = adapter.wrap(enc)
        x = torch.randn(1, 32, 64)
        out = wrapped(x)
        assert out.shape == (1, 32, 64)

    def test_fits_check(self):
        adapter = TMAdapter.auto(verbose=False)
        enc = ToeplitzMagicEncoder(n_layers=2, d_model=64, n_heads=4, seq_len=32)
        result = adapter.fits_in_hardware(enc)
        assert "fits" in result
        assert result["params_B"] > 0
        assert result["compression_ratio"] > 1.0

    def test_check_all_tiers(self, capsys):
        check_all_tiers()
        out = capsys.readouterr().out
        assert "LLaMA" in out
        assert "405B" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
