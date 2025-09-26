"""Hyperboloid manifold placeholder for the JAX backend."""

from __future__ import annotations

from dataclasses import dataclass

from .manifold import Manifold


@dataclass
class Hyperboloid(Manifold):
    """Stub Hyperboloid manifold for Phase 2 work."""

    min_enorm: float = 1e-5
