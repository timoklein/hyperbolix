import torch
import copy

from typing import List


class Manifold(torch.nn.Module):
    """
    Abstract manifold class.
    """
    def __init__(
        self,
        c: torch.Tensor = torch.tensor([1.]),
        trainable_c: bool = False
    ):
        super().__init__()
        if trainable_c:
            self.register_parameter('c', torch.nn.Parameter(c, requires_grad=trainable_c))
        else:
            self.register_buffer('c', c)
        self.min_enorm = None
        self.max_enorm_eps = None
        self.dtype = None

    def _2manifold_dtype(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Convert a list of tensor(s) to the manifold dtype."""
        raise NotImplementedError

    def scalar_mul(self, r: torch.Tensor, x: torch.Tensor, axis: int, backproject: bool) -> torch.Tensor:
        """Multiply manifold point(s) x with scalar(s) r."""
        raise NotImplementedError

    def dist(self, x: torch.Tensor, y: torch.Tensor, axis: int, backproject: bool, version: str) -> torch.Tensor:
        """Compute the geodesic distance(s) between manifold points x and y."""
        raise NotImplementedError

    def dist_0(self, x: torch.Tensor, axis: int, version: str) -> torch.Tensor:
        """Compute the geodesic distance(s) of manifold point(s) x from/to the manifold's origin."""
        raise NotImplementedError

    def expmap(self, v: torch.Tensor, x: torch.Tensor, axis: int, backproject: bool) -> torch.Tensor:
        """Map tangent vector(s) v at manifold point(s) x to the manifold. [Exponential map]"""
        raise NotImplementedError

    def expmap_0(self, v: torch.Tensor, axis: int, backproject: bool) -> torch.Tensor:
        """Map tangent vector(s) v at the manifold's origin to the manifold. [Exponential map]"""
        raise NotImplementedError

    def retraction(self, v: torch.Tensor, x: torch.Tensor, axis: int, backproject: bool) -> torch.Tensor:
        """Approximately map tangent vector(s) v at manifold point(s) x to the manifold. [Retraction map]"""
        raise NotImplementedError

    def logmap(self, y: torch.Tensor, x: torch.Tensor, axis: int, backproject: bool) -> torch.Tensor:
        """Map manifold point(s) y to the tangent space(s) of manifold point(s) x. [Logarithmic map]"""
        raise NotImplementedError

    def logmap_0(self, y: torch.Tensor, axis: int, backproject: bool) -> torch.Tensor:
        """Map manifold point(s) y to the tangent space of the manifold's origin. [Logarithmic map]"""
        raise NotImplementedError

    def ptransp(self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor, axis: int, backproject: bool) -> torch.Tensor:
        """Parallel transport tangent vector(s) v from the tangent space(s) of
           manifold point(s) x to the tangent space(s) of manifold point(s) y."""
        raise NotImplementedError

    def ptransp_0(self, v: torch.Tensor, y: torch.Tensor, axis: int, backproject: bool) -> torch.Tensor:
        """Parallel transport tangent vector(s) v from the tangent space of
           the manifold's origin to the tangent space(s) of manifold point(s) y."""
        raise NotImplementedError

    def tangent_inner(self, u: torch.Tensor, v: torch.Tensor, x: torch.Tensor, axis: int) -> torch.Tensor:
        """Compute the inner product(s) between tangent vectors u and v of the tangent space(s)
           at manifold point(s) x with respect to the Riemannian metric of the manifold."""
        raise NotImplementedError

    def tangent_norm(self, v: torch.Tensor, x: torch.Tensor, axis: int) -> torch.Tensor:
        """Compute the norm(s) of tangent vector(s) v of the tangent space(s) at manifold point(s) x
           with respect to the Riemannian metric of the manifold."""
        raise NotImplementedError

    def egrad2rgrad(self, grad: torch.Tensor, x: torch.Tensor, axis: int) -> torch.Tensor:
        """Compute the Riemannian gradient(s) at manifold point(s) x from the Euclidean gradient(s)."""
        raise NotImplementedError

    def proj(self, x: torch.Tensor, axis: int):
        """Project point(s) x onto the manifold."""
        raise NotImplementedError

    def tangent_proj(self, v: torch.Tensor, x: torch.Tensor, axis: int):
        """Project point(s) v onto the tangent space(s) of manifold point(s) x."""
        raise NotImplementedError

    def is_in_manifold(self, x: torch.Tensor, axis: int) -> bool:
        """Check if point(s) x lie on the manifold."""
        raise NotImplementedError

    def is_in_tangent_space(self, v: torch.Tensor, x: torch.Tensor, axis: int) -> bool:
        """Check if vector(s) v belong to the tangent space(s) at manifold point(s) x."""
        raise NotImplementedError


class ManifoldParameter(torch.nn.Parameter):
    """Subclass of torch.nn.Parameter for Riemannian optimization."""

    def __new__(cls, data: torch.Tensor, requires_grad: bool, manifold: Manifold):
        return torch.nn.Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data: torch.Tensor, requires_grad: bool, manifold: Manifold):
        self.manifold = manifold

    def __repr__(self) -> str:
        return f"{self.manifold.name} Parameter containing:\n" + super(torch.nn.Parameter, self).__repr__()

    def __deepcopy__(self, memo):
        # Deep copy the data
        new_data = copy.deepcopy(self.data, memo)
        # NOTE: Don't deepcopy the manifold because Manifoldparameter never takes ownership of it.
        manifold = self.manifold
        result = ManifoldParameter(new_data, self.requires_grad, manifold)
        return result
