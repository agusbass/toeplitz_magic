"""
matrix.py — TM-Compress core matrix definitions

Matriks 11x11 (HANKEL, bukan Toeplitz):
  val[i][j] = f(i+j)  ← anti-diagonal konstan

  0  11  12  13  14  15  16  17  18  19   0
 11  12  13  14  15  16  17  18  19   0  11
 12  13  14  15  16  17  18  19   0  11  12
 ...

Formula f(s) di mana s = i+j:
  s % 10 == 0  →  0
  1 ≤ s ≤ 9   →  s + 10   (→ 11–19)
  11 ≤ s ≤ 19 →  s        (→ 11–19)

Magic square 3x3:
 8  1  6
 3  5  7
 4  9  2
 → magic constant=15, det=-360, trace=15

Domain disjoint:
 Matrix  ∈ {0, 11, 12, 13, 14, 15, 16, 17, 18, 19}
 Magic   ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}
 ∩ = ∅
"""

import math
import torch


# ─── Seed & lookup ───────────────────────────────────────────────────

TOEPLITZ_SEED = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0]
"""Baris pertama matriks — juga anti-diagonal seed."""

GROUND_TRUTH_11x11 = [
    [ 0, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0],
    [11, 12, 13, 14, 15, 16, 17, 18, 19,  0, 11],
    [12, 13, 14, 15, 16, 17, 18, 19,  0, 11, 12],
    [13, 14, 15, 16, 17, 18, 19,  0, 11, 12, 13],
    [14, 15, 16, 17, 18, 19,  0, 11, 12, 13, 14],
    [15, 16, 17, 18, 19,  0, 11, 12, 13, 14, 15],
    [16, 17, 18, 19,  0, 11, 12, 13, 14, 15, 16],
    [17, 18, 19,  0, 11, 12, 13, 14, 15, 16, 17],
    [18, 19,  0, 11, 12, 13, 14, 15, 16, 17, 18],
    [19,  0, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    [ 0, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0],
]

# Precomputed lookup: index = i+j (0..20), value = matrix value
_HANKEL_LUT = [0] * 21
for _s in range(21):
    if _s % 10 == 0:
        _HANKEL_LUT[_s] = 0
    elif _s <= 9:
        _HANKEL_LUT[_s] = _s + 10
    else:
        _HANKEL_LUT[_s] = _s % 20 if _s % 20 != 0 else 0

MAGIC_KERNEL_LIST = [
    [8, 1, 6],
    [3, 5, 7],
    [4, 9, 2],
]


# ─── Builders ────────────────────────────────────────────────────────

def build_toeplitz_11x11(verify: bool = True) -> torch.Tensor:
    """
    Bangun matriks 11×11.
    Formula: val[i][j] = f(i+j)  — Hankel (anti-diagonal konstan)

    Args:
        verify: cocokkan dengan ground truth

    Returns:
        Tensor (11, 11) float32
    """
    n = 11
    i_idx = torch.arange(n)
    j_idx = torch.arange(n)
    s = i_idx.unsqueeze(1) + j_idx.unsqueeze(0)   # (11,11) sum matrix
    lut = torch.tensor(_HANKEL_LUT, dtype=torch.float32)
    T = lut[s]

    if verify:
        gt = torch.tensor(GROUND_TRUTH_11x11, dtype=torch.float32)
        assert torch.allclose(T, gt), \
            f"Tidak cocok ground truth!\nBuilt:\n{T.int()}\nGT:\n{gt.int()}"

    return T


def build_magic_kernel(verify: bool = True) -> torch.Tensor:
    """
    Magic square 3×3. Magic constant=15, det=-360, trace=15.

    Returns:
        Tensor (3, 3) float32
    """
    M = torch.tensor(MAGIC_KERNEL_LIST, dtype=torch.float32)

    if verify:
        for i in range(3):
            assert M[i].sum().item() == 15,         f"Baris {i} != 15"
            assert M[:, i].sum().item() == 15,      f"Kolom {i} != 15"
        assert M.diag().sum().item() == 15,         "Diagonal utama != 15"
        assert M.flip(1).diag().sum().item() == 15, "Anti-diagonal != 15"
        det = torch.linalg.det(M.double()).item()
        assert abs(det - (-360)) < 1e-6,            f"Det={det}, bukan -360"

    return M


MAGIC_KERNEL = build_magic_kernel(verify=False)


def build_toeplitz_bias(seq_len: int) -> torch.Tensor:
    """
    Ekspansi ke bias matriks seq_len×seq_len.
    bias[i][j] = f((i+j) % 20) / 19.0  — periodik, range [0,1]

    Digunakan sebagai positional bias di attention.
    Fixed — tidak perlu ditraining.

    Returns:
        Tensor (seq_len, seq_len) float32
    """
    lut = torch.tensor(_HANKEL_LUT, dtype=torch.float32)
    i_idx = torch.arange(seq_len)
    j_idx = torch.arange(seq_len)
    s = (i_idx.unsqueeze(1) + j_idx.unsqueeze(0)) % 20
    bias = lut[s] / 19.0
    return bias


def build_dual_domain_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    Dual-domain positional encoding.

    Domain A — posisi  {0,11–19} dari matriks → separuh d_model pertama
    Domain B — semantik {1–9}   dari Magic    → separuh d_model kedua
    Domain A ∩ Domain B = ∅

    Returns:
        Tensor (seq_len, d_model) float32
    """
    T = build_toeplitz_11x11(verify=False)
    M = MAGIC_KERNEL
    half = d_model // 2

    toep_row = T[0] / 19.0
    pos_enc  = toep_row.repeat(math.ceil(seq_len / 11))[:seq_len]
    pos_ch   = pos_enc.unsqueeze(1).expand(-1, half)

    magic_flat = M.flatten() / 9.0
    sem_enc    = magic_flat.repeat(math.ceil(seq_len / 9))[:seq_len]
    sem_ch     = sem_enc.unsqueeze(1).expand(-1, d_model - half)

    return torch.cat([pos_ch, sem_ch], dim=1)


# ─── Verification helpers ─────────────────────────────────────────────

def verify_disjoint() -> bool:
    """Pastikan domain matriks dan Magic Square tidak bertabrakan."""
    T = build_toeplitz_11x11(verify=False)
    M = MAGIC_KERNEL
    return len(set(T.flatten().int().tolist()) &
               set(M.flatten().int().tolist())) == 0


def print_report():
    """Cetak laporan lengkap properti kedua matriks."""
    T = build_toeplitz_11x11()
    M = build_magic_kernel()

    print("=" * 56)
    print("Matriks 11×11 — Hankel (anti-diagonal konstan)")
    print("=" * 56)
    print(T.int().numpy())
    print(f"Formula       : val[i][j] = f(i+j)")
    print(f"Simetris      : {torch.allclose(T, T.T)}")
    print(f"Nilai unik    : {sorted(set(T.flatten().int().tolist()))}")
    print(f"Range         : [{int(T.min())}, {int(T.max())}]")

    print()
    print("=" * 56)
    print("Magic square 3×3")
    print("=" * 56)
    print(M.int().numpy())
    det = torch.linalg.det(M.double()).item()
    print(f"Magic const   : {int(M[0].sum())}")
    print(f"Determinant   : {det:.1f}")
    print(f"Trace         : {int(M.diag().sum())}")
    print(f"Eigenvalues   : {torch.linalg.eigvals(M.double()).numpy()}")

    print()
    print("=" * 56)
    print("Domain disjoint check")
    print("=" * 56)
    t_vals  = sorted(set(T.flatten().int().tolist()))
    m_vals  = sorted(set(M.flatten().int().tolist()))
    union   = sorted(set(t_vals + m_vals))
    missing = [x for x in range(0, 20) if x not in union]
    print(f"T  domain : {t_vals}")
    print(f"MS domain : {m_vals}")
    print(f"Disjoint  : {verify_disjoint()}")
    print(f"Union     : {union}")
    print(f"Missing   : {missing}  ← null/anchor token")


if __name__ == "__main__":
    print_report()
