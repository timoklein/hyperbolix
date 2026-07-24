"""Hyperbolic dimensionality reduction and decomposition.

Provides HoroPCA (Chami et al. 2021) — hyperbolic PCA via horospherical projections — CO-SNE
(Guo et al. 2022) — hyperbolic t-SNE with magnitude (distance-to-origin) preservation — and
the Fréchet-mean data-centering primitive HoroPCA builds on.
"""

from .cosne import (
    CoSNE,
    conditional_probabilities,
    fit_cosne,
    joint_probabilities,
    kl_divergence_loss,
    low_dim_probabilities,
    magnitude_loss,
)
from .frechet import frechet_mean
from .horopca import (
    HoroPCA,
    fit_horopca,
    horo_projection,
    horopca_loss,
    transform_horopca,
)

__all__ = [
    "CoSNE",
    "HoroPCA",
    "conditional_probabilities",
    "fit_cosne",
    "fit_horopca",
    "frechet_mean",
    "horo_projection",
    "horopca_loss",
    "joint_probabilities",
    "kl_divergence_loss",
    "low_dim_probabilities",
    "magnitude_loss",
    "transform_horopca",
]
