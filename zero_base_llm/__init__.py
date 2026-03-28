"""
Zero-Base LLM: A lightweight LLM built from binary foundation.

This package implements a 22-layer LLM architecture starting from
binary (0/1) foundation through ASCII encoding up to paragraph
generation with bidirectional self-study.
"""

from .config import ZeroBaseConfig
from .model.model import ZeroBaseLLM

__version__ = "0.1.0"
__all__ = ["ZeroBaseConfig", "ZeroBaseLLM"]