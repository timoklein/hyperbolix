"""Hyperbolic dimensionality reduction and decomposition.

Provides HoroPCA (Chami et al. 2021) — hyperbolic PCA via horospherical projections — and
the Fréchet-mean data-centering primitive it builds on.
"""

from .frechet import frechet_mean
from .horopca import (
    HoroPCA,
    fit_horopca,
    horo_projection,
    horopca_loss,
    transform_horopca,
)

__all__ = [
    "HoroPCA",
    "fit_horopca",
    "frechet_mean",
    "horo_projection",
    "horopca_loss",
    "transform_horopca",
]
