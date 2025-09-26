"""This file contains global fixtures that are used across all our tests."""

import pytest
import torch

from typing import Tuple
from src.manifolds import Manifold, Euclidean, Hyperboloid, PoincareBall


@pytest.fixture(scope="package", params=[*range(10, 13)])
def seed(request: pytest.FixtureRequest) -> None:
    """Global seed for reproducibility."""
    torch.manual_seed(request.param)

@pytest.fixture(scope="package", params=["float32", "float64"])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Test different data types."""
    return request.param

@pytest.fixture(scope="package")
def tolerance(dtype: str) -> Tuple[float, float]:
    """Set numerical tolerances for floating point comparisons."""
    if dtype == "float32":
        atol = 4e-03
        rtol = 4e-03
    else:   # float64
        atol = 1e-07
        rtol = 1e-07
    return atol, rtol

@pytest.fixture(scope="package", params=[Euclidean, Hyperboloid, PoincareBall], ids=["Euclidean", "Hyperboloid", "PoincareBall"])
def manifold(seed: None, dtype: str, request: pytest.FixtureRequest) -> Manifold:
    """Test different manifolds and curvatures."""
    if dtype == "float32":
        c_dtype = torch.float32
    else:   # float64
        c_dtype = torch.float64
    c = torch.empty(1, dtype=c_dtype).exponential_(0.5)
    return request.param(c=c, dtype=dtype)

@pytest.fixture(scope="package", params=[2, 5, 10, 15])
def uniform_points(seed: None, manifold: Manifold, request: pytest.FixtureRequest) -> torch.Tensor:
    """Helper to generate uniformly distributed points for each manifold type."""
    dim = request.param
    num_pts = 2_500 * 6
    if isinstance(manifold, Euclidean):
        bound = 100
        points = torch.empty((num_pts, dim), dtype=manifold.dtype).uniform_(-bound, bound)
    else:   # PoincareBall & Hyperboloid
        random_dirs = torch.normal(0, 1, size=(num_pts, dim), dtype=manifold.dtype)
        random_dirs /= random_dirs.norm(p=2, dim=-1, keepdim=True)
        random_radii = torch.rand((num_pts, 1), dtype=manifold.dtype).pow(1 / dim)
        points = manifold.c**-0.5 * (random_dirs * random_radii)
        if isinstance(manifold, Hyperboloid): # Hyperboloid
            poincare = PoincareBall(c=manifold.c, dtype=manifold.dtype)
            # Scale the points to account for the representational limitations of the Hyperboloid
            points = points * 0.5
            points = poincare.to_hyperboloid(points)
            points = manifold.proj(points)
    # Check if the points are in the manifold
    assert manifold.is_in_manifold(points), "Points are not in manifold!"
    return points
