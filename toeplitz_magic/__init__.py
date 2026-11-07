from .matrix import build_toeplitz_11x11, build_dual_domain_encoding, MAGIC_KERNEL, TOEPLITZ_SEED
from .attention import ToeplitzMagicAttention, ToeplitzMagicBlock, ToeplitzMagicEncoder
from .tier_adapter import TMAdapter, StreamingToeplitzBias, detect_hardware, check_all_tiers
from .hf_adapter import ToeplitzHFAdapter, LlamaToeplitzAttention

__all__ = [
    # matrix
    "build_toeplitz_11x11",
    "build_dual_domain_encoding",
    "MAGIC_KERNEL",
    "TOEPLITZ_SEED",
    # attention
    "ToeplitzMagicAttention",
    "ToeplitzMagicBlock",
    "ToeplitzMagicEncoder",
    # tier adapter
    "TMAdapter",
    "StreamingToeplitzBias",
    "detect_hardware",
    "check_all_tiers",
    # hf adapter
    "ToeplitzHFAdapter",
    "LlamaToeplitzAttention",
]
__version__ = "0.3.0"
