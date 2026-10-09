"""
Fractional cascading index.

A layered set of sorted arrays sharing sampled elements between adjacent
layers to answer multi-level binary searches in a single logarithmic pass.
"""

from .core import (
    Cascade,
    search_all,
)

__all__ = ["Cascade", "search_all"]
