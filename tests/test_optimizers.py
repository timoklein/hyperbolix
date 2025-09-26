import pytest
import torch

from typing import Tuple
from src.manifolds import Manifold, ManifoldParameter
from src.optim import RiemannianAdam, RiemannianSGD


@pytest.mark.parametrize("expmap_update", [True, False])
def test_riemannian_adam(manifold: Manifold, tolerance: Tuple[float, float],
                         uniform_points: torch.Tensor, expmap_update: bool) -> None:
    """Test the RiemannianAdam for convergence."""
    _, rtol = tolerance
    target = uniform_points[0, :]
    start = manifold.scalar_mul(torch.tensor([0.9]), target)
    start = ManifoldParameter(start, requires_grad=True, manifold=manifold)

    optim = RiemannianAdam([start], lr=1e-3, eps=1e-5, expmap_update=expmap_update, backproject=True)
    for _ in range(300_000):
        optim.zero_grad()
        loss = manifold.dist(start, target).pow(2).mean()
        if loss < rtol:
            break
        loss.backward()
        optim.step()
    else:
        assert False, "RiemannianAdam did not converge!"

@pytest.mark.parametrize("expmap_update", [True, False])
def test_riemannian_sgd(manifold: Manifold, tolerance: Tuple[float, float],
                        uniform_points: torch.Tensor, expmap_update: bool) -> None:
    """Test the RiemannianSGD for convergence."""
    _, rtol = tolerance
    target = uniform_points[0, :]
    start = manifold.scalar_mul(torch.tensor([0.3]), target)
    start = ManifoldParameter(start, requires_grad=True, manifold=manifold)

    optim = RiemannianSGD([start], lr=1e-3, momentum=0.9, expmap_update=expmap_update, backproject=True)
    for _ in range(1000):
        optim.zero_grad()
        loss = manifold.dist(start, target).pow(2).mean()
        if loss < rtol:
            break
        loss.backward()
        optim.step()
    else:
        assert False, "RiemannianSGD did not converge!"
