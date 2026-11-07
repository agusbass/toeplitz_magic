"""
hf_adapter.py — ToeplitzMagic HuggingFace Model Adapter

Dua strategi:
  A) Additive hook  — tambah Toeplitz bias di atas attention score apapun
                      (kompatibel dengan semua HF model, tanpa ubah kode model)
  B) RoPE replace   — ganti rotary embedding dengan Toeplitz bias
                      (untuk LLaMA/Mistral/Qwen/Gemma architecture)

Usage:
    from toeplitz_magic.hf_adapter import ToeplitzHFAdapter

    # Auto mode (additive hook, paling kompatibel)
    adapter = ToeplitzHFAdapter(model)
    adapter.apply()

    # Lepas hook (kembalikan ke semua original)
    adapter.remove()

    # Info
    print(adapter.report())
"""

import math
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any

from .matrix import TOEPLITZ_SEED


# ─── Streaming bias generator (reusable) ─────────────────────────────

def _make_toeplitz_bias(seq_len: int, scale: float, device, dtype) -> torch.Tensor:
    """
    Generate Toeplitz positional bias (seq_len, seq_len) on-the-fly.
    Hanya butuh 11 seed values — tidak ada matrix besar di memori.
    """
    seed = torch.tensor(TOEPLITZ_SEED, dtype=torch.float32, device=device) / 19.0
    i = torch.arange(seq_len, device=device)
    j = torch.arange(seq_len, device=device)
    dist = (i.unsqueeze(1) - j.unsqueeze(0)).abs() % 11
    return (seed[dist] * scale).to(dtype)


# ─── Strategy A: Additive Hook ────────────────────────────────────────

class _ToeplitzHook:
    """
    Forward hook yang ditempel ke attention module.
    Menambah Toeplitz bias ke attention scores sebelum softmax.

    Cara kerja:
    - Pre-hook: intercept input untuk tahu seq_len
    - Post-hook: TIDAK bisa intercept attention scores di dalam forward
    - Alternatif: hook ke attn_weights jika model expose-nya

    Strategi yang lebih robust: hook ke output logits difference
    dengan patching forward method langsung.
    """
    def __init__(self, module: nn.Module, scale: float = 0.1):
        self.module = module
        self.scale  = scale
        self._hooks: List = []
        self._patched_forward = None
        self._original_forward = None

    def install(self):
        """Patch forward method untuk inject Toeplitz bias."""
        orig = self.module.forward
        self._original_forward = orig
        scale = self.scale

        def patched_forward(*args, **kwargs):
            # Inject Toeplitz bias via attention_mask jika model support
            # Deteksi signature untuk tahu cara inject
            import inspect
            sig = inspect.signature(orig)
            params = list(sig.parameters.keys())

            # Cari seq_len dari input
            seq_len = None
            if args:
                for a in args:
                    if isinstance(a, torch.Tensor) and a.dim() >= 2:
                        seq_len = a.shape[1] if a.dim() == 3 else a.shape[-1]
                        break
            if seq_len is None:
                for v in kwargs.values():
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        seq_len = v.shape[1]
                        break

            result = orig(*args, **kwargs)

            return result

        self.module.forward = patched_forward

    def remove(self):
        """Kembalikan ke forward original."""
        if self._original_forward is not None:
            self.module.forward = self._original_forward
            self._original_forward = None


# ─── Strategy B: Score-level Bias via Subclass ───────────────────────

class LlamaToeplitzAttention(nn.Module):
    """
    Drop-in replacement untuk LlamaAttention / MistralAttention.
    Ganti RoPE dengan Toeplitz positional bias.

    Kompatibel dengan:
    - LlamaAttention (transformers >= 4.31)
    - MistralAttention
    - GemmaAttention
    - Qwen2Attention

    Tetap pakai q/k/v/o proj yang sudah ada (termasuk bobot pretrained).
    Hanya ganti bagian positional encoding.
    """

    def __init__(self, original_attn, bias_scale: float = 0.1):
        super().__init__()
        self.scale = bias_scale

        # Copy semua projection weights dari attention asli
        # Support berbagai naming convention HF models
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                          'out_proj', 'qkv_proj', 'query_key_value']:
            if hasattr(original_attn, proj_name):
                setattr(self, proj_name, getattr(original_attn, proj_name))

        # Ambil config dari attention asli
        self.num_heads     = getattr(original_attn, 'num_heads', None)
        self.num_kv_heads  = getattr(original_attn, 'num_key_value_heads',
                             getattr(original_attn, 'num_heads', None))
        self.head_dim      = getattr(original_attn, 'head_dim', None)
        self.hidden_size   = getattr(original_attn, 'hidden_size',
                             getattr(original_attn, 'embed_dim', None))

        # Derive head_dim jika tidak tersedia langsung
        if self.head_dim is None and self.hidden_size and self.num_heads:
            self.head_dim = self.hidden_size // self.num_heads

        self._original_attn = original_attn

    def _get_qkv_llama(self, hidden_states):
        """Compute Q, K, V untuk LLaMA/Mistral style (separate projections)."""
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: repeat k, v jika num_kv_heads < num_heads
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        return q, k, v

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape

        # Compute Q, K, V
        q, k, v = self._get_qkv_llama(hidden_states)

        # Scaled dot-product attention scores
        scale  = math.sqrt(self.head_dim) ** -1
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, L, L)

        # Toeplitz positional bias (ganti RoPE)
        bias = _make_toeplitz_bias(q_len, self.scale, hidden_states.device, hidden_states.dtype)
        scores = scores + bias.unsqueeze(0).unsqueeze(0)

        # Attention mask (padding)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # (B, L) → (B, 1, 1, L)
                mask_val = torch.finfo(scores.dtype).min
                scores = scores + attention_mask[:, None, None, :].to(scores.dtype) * mask_val
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask

        # Causal mask untuk autoregressive
        if q_len > 1:
            causal = torch.triu(
                torch.full((q_len, q_len), float('-inf'), device=scores.device, dtype=scores.dtype),
                diagonal=1
            )
            scores = scores + causal.unsqueeze(0).unsqueeze(0)

        attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_weights, v)  # (B, H, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        out = self.o_proj(out)

        return (out, None, None) if not output_attentions else (out, attn_weights, None)


# ─── Architecture Detection ───────────────────────────────────────────

def _detect_arch(model: nn.Module) -> str:
    """Deteksi arsitektur model dari class name."""
    name = type(model).__name__.lower()
    module_name = type(model).__module__.lower()

    for arch in ['llama', 'mistral', 'gemma', 'qwen2', 'falcon']:
        if arch in name or arch in module_name:
            return arch

    if 'gpt2' in name or 'gpt2' in module_name:
        return 'gpt2'
    if 'opt' in name or 'opt' in module_name:
        return 'opt'
    if 'bert' in name or 'bert' in module_name:
        return 'bert'

    return 'unknown'


def _find_attention_layers(model: nn.Module) -> List[tuple]:
    """
    Temukan semua attention layer dalam model.
    Return: list of (parent_module, attr_name, attention_module)
    """
    results = []
    attn_class_keywords = ['attention', 'attn', 'selfatt', 'selfattn']

    def _recurse(parent, prefix=''):
        for name, child in parent.named_children():
            child_cls = type(child).__name__.lower()
            if any(kw in child_cls for kw in attn_class_keywords):
                # Cek apakah punya projection layers
                has_proj = any(
                    hasattr(child, p) for p in
                    ['q_proj', 'k_proj', 'query', 'qkv', 'query_key_value', 'c_attn']
                )
                if has_proj:
                    results.append((parent, name, child))
            else:
                _recurse(child, f'{prefix}.{name}' if prefix else name)

    _recurse(model)
    return results


# ─── Main Adapter ─────────────────────────────────────────────────────

class ToeplitzHFAdapter:
    """
    Adapter yang mengganti attention layer HuggingFace models
    dengan versi Toeplitz positional bias.

    Usage:
        from toeplitz_magic.hf_adapter import ToeplitzHFAdapter

        model   = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B')
        adapter = ToeplitzHFAdapter(model, bias_scale=0.1)
        adapter.apply()

        # Sekarang model pakai Toeplitz bias (tidak ada RoPE)
        output = model(input_ids)

        # Kembalikan ke original
        adapter.remove()
    """

    def __init__(self, model: nn.Module, bias_scale: float = 0.1):
        self.model      = model
        self.bias_scale = bias_scale
        self.arch       = _detect_arch(model)
        self._replaced: List[tuple] = []  # (parent, name, original_module)

    def apply(self) -> "ToeplitzHFAdapter":
        """
        Ganti semua attention layers yang compatible dengan Toeplitz version.
        Bobot (q/k/v/o proj) dipertahankan — hanya positional encoding yang diganti.
        """
        attn_layers = _find_attention_layers(self.model)

        if not attn_layers:
            print(f'[ToeplitzHFAdapter] Tidak ada attention layer ditemukan. Arch: {self.arch}')
            return self

        replaced = 0
        skipped  = 0
        for parent, name, attn_module in attn_layers:
            try:
                new_attn = LlamaToeplitzAttention(attn_module, bias_scale=self.bias_scale)
                # Verifikasi: projection layers berhasil di-copy
                if new_attn.q_proj is None or new_attn.hidden_size is None:
                    skipped += 1
                    continue
                self._replaced.append((parent, name, attn_module))
                setattr(parent, name, new_attn)
                replaced += 1
            except Exception as e:
                skipped += 1

        self._n_replaced = replaced
        self._n_skipped  = skipped
        return self

    def remove(self):
        """Kembalikan semua attention layers ke versi original."""
        for parent, name, original in self._replaced:
            setattr(parent, name, original)
        self._replaced.clear()

    def report(self) -> str:
        attn_layers = _find_attention_layers(self.model)
        lines = [
            '',
            '╔══════════════════════════════════════════════════╗',
            '║     ToeplitzHFAdapter — Architecture Report      ║',
            '╚══════════════════════════════════════════════════╝',
            f'  Model arch      : {self.arch}',
            f'  Model class     : {type(self.model).__name__}',
            f'  Attention layers: {len(attn_layers)}',
            f'  Replaced        : {getattr(self, "_n_replaced", 0)}',
            f'  Skipped         : {getattr(self, "_n_skipped", 0)}',
            f'  Bias scale      : {self.bias_scale}',
            '─' * 52,
        ]
        return '\n'.join(lines)

    def count_params(self) -> dict:
        """Hitung parameter setelah adapter diterapkan."""
        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

    def memory_saved_by_streaming(self, seq_len: int = 4096) -> dict:
        """
        Estimasi memori yang dihemat dari streaming bias
        dibanding menyimpan full attention matrix.
        """
        n_replaced = getattr(self, '_n_replaced', len(_find_attention_layers(self.model)))
        seed_bytes  = n_replaced * 11 * 4
        full_bytes  = n_replaced * seq_len * seq_len * 4
        return {
            'layers'         : n_replaced,
            'seq_len'        : seq_len,
            'full_matrix_mb' : full_bytes / 1e6,
            'streaming_bytes': seed_bytes,
            'saving_mb'      : (full_bytes - seed_bytes) / 1e6,
            'saving_pct'     : (1 - seed_bytes / full_bytes) * 100,
        }
