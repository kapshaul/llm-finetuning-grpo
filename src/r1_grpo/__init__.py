"""Inspectable pure-PyTorch study implementation of the DeepSeek-R1 (v1) GRPO pipeline.

The package root is intentionally lightweight (no torch import) so that
``r1-grpo --help`` stays fast. Import submodules explicitly, for example
``from r1_grpo.core import grpo_loss``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
