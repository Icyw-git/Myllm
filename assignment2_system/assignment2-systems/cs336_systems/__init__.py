import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336-systems")
except importlib.metadata.PackageNotFoundError:
    pass

# Convenience re-exports for the assignment's public API.
from .attention import FlashAttentionPytorchAutogradFunction, FlashAttentionTritonAutogradFunction
from .distributed import DDP, FSDP
from .optimizer import ShardedOptimizer

__all__ = [
    "DDP",
    "FSDP",
    "FlashAttentionPytorchAutogradFunction",
    "FlashAttentionTritonAutogradFunction",
    "ShardedOptimizer",
]
