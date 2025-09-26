import torch

from ..manifolds import Manifold, Hyperboloid


class Expmap(torch.nn.Module):
    """
    Module to compute the exponential map at a point on the manifold.

    Parameters
    ----------
    manifold : Manifold
        The hyperbolic manifold
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1)
    backproject : bool
        Whether to project results back to the manifold (default: True)
    """
    def __init__(self, manifold: Manifold, hyperbolic_axis: int=-1, backproject: bool=True):
        super().__init__()
        self.manifold = manifold
        self.hyperbolic_axis = hyperbolic_axis
        self.backproject = backproject

    def forward(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.manifold.expmap(v, x, axis=self.hyperbolic_axis, backproject=self.backproject)

class Expmap_0(torch.nn.Module):
    """
    Module to compute the exponential map at the origin of the manifold.

    Parameters
    ----------
    manifold : Manifold
        The hyperbolic manifold
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1)
    backproject : bool
        Whether to project results back to the manifold (default: True)
    """
    def __init__(self, manifold: Manifold, hyperbolic_axis: int=-1, backproject: bool=True):
        super().__init__()
        self.manifold = manifold
        self.hyperbolic_axis = hyperbolic_axis
        self.backproject = backproject

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.manifold.expmap_0(v, axis=self.hyperbolic_axis, backproject=self.backproject)

class Retraction(torch.nn.Module):
    """
    Module to compute the retraction map at a point on the manifold.

    Parameters
    ----------
    manifold : Manifold
        The hyperbolic manifold
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1)
    backproject : bool
        Whether to project results back to the manifold (default: True)
    """
    def __init__(self, manifold: Manifold, hyperbolic_axis: int=-1, backproject: bool=True):
        super().__init__()
        self.manifold = manifold
        self.hyperbolic_axis = hyperbolic_axis
        self.backproject = backproject

    def forward(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.manifold.retraction(v, x, axis=self.hyperbolic_axis, backproject=self.backproject)

class Logmap(torch.nn.Module):
    """
    Module to compute the logarithmic map at a point on the manifold.

    Parameters
    ----------
    manifold : Manifold
        The hyperbolic manifold
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1)
    backproject : bool
        Whether to project results back to the tangent space (default: True)
    """
    def __init__(self, manifold: Manifold, hyperbolic_axis: int=-1, backproject: bool=True):
        super().__init__()
        self.manifold = manifold
        self.hyperbolic_axis = hyperbolic_axis
        self.backproject = backproject

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.manifold.logmap(y, x, axis=self.hyperbolic_axis, backproject=self.backproject)

class Logmap_0(torch.nn.Module):
    """
    Module to compute the logarithmic map at the origin of the manifold.

    Parameters
    ----------
    manifold : Manifold
        The hyperbolic manifold
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1)
    backproject : bool
        Whether to project results back to the tangent space (default: True)
    """
    def __init__(self, manifold: Manifold, hyperbolic_axis: int=-1, backproject: bool=True):
        super().__init__()
        self.manifold = manifold
        self.hyperbolic_axis = hyperbolic_axis
        self.backproject = backproject

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.manifold.logmap_0(y, axis=self.hyperbolic_axis, backproject=self.backproject)

class Proj(torch.nn.Module):
    """
    Module to compute the (back)projection onto the manifold to account for numerical instabilities.

    Parameters
    ----------
    manifold : Manifold
        The hyperbolic manifold
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1)
    """
    def __init__(self, manifold: Manifold, hyperbolic_axis: int=-1):
        super().__init__()
        self.manifold = manifold
        self.hyperbolic_axis = hyperbolic_axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.manifold.proj(x, axis=self.hyperbolic_axis)

class TanProj(torch.nn.Module):
    """
    Module to compute the (back)projection onto the tangent space
    at a point on the manifold to account for numerical instabilities.

    Parameters
    ----------
    manifold : Manifold
        The hyperbolic manifold
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1)
    """
    def __init__(self, manifold: Manifold, hyperbolic_axis: int=-1):
        super().__init__()
        self.manifold = manifold
        self.hyperbolic_axis = hyperbolic_axis

    def forward(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.manifold.tangent_proj(v, x, axis=self.hyperbolic_axis)

class HyperbolicActivation(torch.nn.Module):
    """
    Module to apply an activation function in the tangent space at the manifold origin.

    Parameters
    ----------
    manifold : Manifold
        The hyperbolic manifold
    activation : torch.nn.Module
        The activation function to apply in the tangent space at the manifold origin
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1)
    backproject : bool
        Whether to project results back to the manifold (default: True)
    """
    def __init__(self, manifold: Manifold, activation: torch.nn.Module, hyperbolic_axis: int=-1, backproject: bool=True):
        super().__init__()
        self.manifold = manifold
        self.hyperbolic_axis = hyperbolic_axis
        self.backproject = backproject
        if isinstance(self.manifold, Hyperboloid):
            assert activation(torch.tensor(0.)) == torch.tensor(0.), \
                "The Hyperboloid activation must map 0 to 0 to map tangent vectors of the manifold origin to tangent vectors of the manifold origin"
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.manifold.logmap_0(x, axis=self.hyperbolic_axis, backproject=self.backproject)
        v = self.activation(v)
        return self.manifold.expmap_0(v, axis=self.hyperbolic_axis, backproject=self.backproject)
