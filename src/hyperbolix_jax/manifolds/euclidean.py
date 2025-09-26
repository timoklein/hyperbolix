"""Euclidean manifold placeholder for the JAX backend."""

from __future__ import annotations

from dataclasses import dataclass

from .manifold import Manifold


@dataclass
class Euclidean(Manifold):
    """Stub Euclidean manifold; implementations land in Phase 2 port."""

    pass
