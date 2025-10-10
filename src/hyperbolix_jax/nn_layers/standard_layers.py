"""Standard hyperbolic neural network layers (wrappers around manifold operations).

These layers are designed to be JIT-compatible with nnx.jit. All configuration parameters
(manifold_module, hyperbolic_axis, backproject) are treated as static and will be baked
into the compiled function. Changing these values after JIT compilation will trigger
automatic recompilation.
"""

from typing import Any

from flax import nnx
from jaxtyping import Array, Float


class Expmap(nnx.Module):
    """
    Module to compute the exponential map at a point on the manifold.

    Parameters
    ----------
    manifold_module : module
        The hyperbolic manifold module (e.g., poincare, hyperboloid)
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1).
        Note: This is a static configuration.
    backproject : bool
        Whether to project results back to the manifold (default: True).
        Note: This is a static configuration.

    Notes
    -----
    JIT Compatibility:
        This layer is fully compatible with nnx.jit.
    """

    def __init__(self, manifold_module: Any, hyperbolic_axis: int = -1, backproject: bool = True):
        # Static configuration (treated as compile-time constants for JIT)
        self.manifold = manifold_module
        self.hyperbolic_axis = hyperbolic_axis
        self.backproject = backproject

    def __call__(self, v: Float[Array, "..."], x: Float[Array, "..."], c: float = 1.0) -> Float[Array, "..."]:
        """Compute exponential map at point x."""
        return self.manifold.expmap(v, x, c, axis=self.hyperbolic_axis, backproject=self.backproject)


class Expmap0(nnx.Module):
    """
    Module to compute the exponential map at the origin of the manifold.

    Parameters
    ----------
    manifold_module : module
        The hyperbolic manifold module (e.g., poincare, hyperboloid)
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1).
        Note: This is a static configuration.
    backproject : bool
        Whether to project results back to the manifold (default: True).
        Note: This is a static configuration.

    Notes
    -----
    JIT Compatibility:
        This layer is fully compatible with nnx.jit.
    """

    def __init__(self, manifold_module: Any, hyperbolic_axis: int = -1, backproject: bool = True):
        # Static configuration (treated as compile-time constants for JIT)
        self.manifold = manifold_module
        self.hyperbolic_axis = hyperbolic_axis
        self.backproject = backproject

    def __call__(self, v: Float[Array, "..."], c: float = 1.0) -> Float[Array, "..."]:
        """Compute exponential map at origin."""
        return self.manifold.expmap_0(v, c, axis=self.hyperbolic_axis, backproject=self.backproject)


class Retraction(nnx.Module):
    """
    Module to compute the retraction map at a point on the manifold.

    Parameters
    ----------
    manifold_module : module
        The hyperbolic manifold module (e.g., poincare, hyperboloid)
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1).
        Note: This is a static configuration.
    backproject : bool
        Whether to project results back to the manifold (default: True).
        Note: This is a static configuration.

    Notes
    -----
    JIT Compatibility:
        This layer is fully compatible with nnx.jit.
    """

    def __init__(self, manifold_module: Any, hyperbolic_axis: int = -1, backproject: bool = True):
        # Static configuration (treated as compile-time constants for JIT)
        self.manifold = manifold_module
        self.hyperbolic_axis = hyperbolic_axis
        self.backproject = backproject

    def __call__(self, v: Float[Array, "..."], x: Float[Array, "..."], c: float = 1.0) -> Float[Array, "..."]:
        """Compute retraction at point x."""
        return self.manifold.retraction(v, x, c, axis=self.hyperbolic_axis, backproject=self.backproject)


class Logmap(nnx.Module):
    """
    Module to compute the logarithmic map at a point on the manifold.

    Parameters
    ----------
    manifold_module : module
        The hyperbolic manifold module (e.g., poincare, hyperboloid)
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1).
        Note: This is a static configuration.
    backproject : bool
        Whether to project results back to the tangent space (default: True).
        Note: This is a static configuration.

    Notes
    -----
    JIT Compatibility:
        This layer is fully compatible with nnx.jit.
    """

    def __init__(self, manifold_module: Any, hyperbolic_axis: int = -1, backproject: bool = True):
        # Static configuration (treated as compile-time constants for JIT)
        self.manifold = manifold_module
        self.hyperbolic_axis = hyperbolic_axis
        self.backproject = backproject

    def __call__(self, y: Float[Array, "..."], x: Float[Array, "..."], c: float = 1.0) -> Float[Array, "..."]:
        """Compute logarithmic map at point x."""
        return self.manifold.logmap(y, x, c, axis=self.hyperbolic_axis, backproject=self.backproject)


class Logmap0(nnx.Module):
    """
    Module to compute the logarithmic map at the origin of the manifold.

    Parameters
    ----------
    manifold_module : module
        The hyperbolic manifold module (e.g., poincare, hyperboloid)
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1).
        Note: This is a static configuration.
    backproject : bool
        Whether to project results back to the tangent space (default: True).
        Note: This is a static configuration.

    Notes
    -----
    JIT Compatibility:
        This layer is fully compatible with nnx.jit.
    """

    def __init__(self, manifold_module: Any, hyperbolic_axis: int = -1, backproject: bool = True):
        # Static configuration (treated as compile-time constants for JIT)
        self.manifold = manifold_module
        self.hyperbolic_axis = hyperbolic_axis
        self.backproject = backproject

    def __call__(self, y: Float[Array, "..."], c: float = 1.0) -> Float[Array, "..."]:
        """Compute logarithmic map at origin."""
        return self.manifold.logmap_0(y, c, axis=self.hyperbolic_axis, backproject=self.backproject)


class Proj(nnx.Module):
    """
    Module to compute the (back)projection onto the manifold to account for numerical instabilities.

    Parameters
    ----------
    manifold_module : module
        The hyperbolic manifold module (e.g., poincare, hyperboloid)
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1).
        Note: This is a static configuration.

    Notes
    -----
    JIT Compatibility:
        This layer is fully compatible with nnx.jit.
    """

    def __init__(self, manifold_module: Any, hyperbolic_axis: int = -1):
        # Static configuration (treated as compile-time constants for JIT)
        self.manifold = manifold_module
        self.hyperbolic_axis = hyperbolic_axis

    def __call__(self, x: Float[Array, "..."], c: float = 1.0) -> Float[Array, "..."]:
        """Project onto manifold."""
        return self.manifold.proj(x, c, axis=self.hyperbolic_axis)


class TanProj(nnx.Module):
    """
    Module to compute the (back)projection onto the tangent space
    at a point on the manifold to account for numerical instabilities.

    Parameters
    ----------
    manifold_module : module
        The hyperbolic manifold module (e.g., poincare, hyperboloid)
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1).
        Note: This is a static configuration.

    Notes
    -----
    JIT Compatibility:
        This layer is fully compatible with nnx.jit.
    """

    def __init__(self, manifold_module: Any, hyperbolic_axis: int = -1):
        # Static configuration (treated as compile-time constants for JIT)
        self.manifold = manifold_module
        self.hyperbolic_axis = hyperbolic_axis

    def __call__(self, v: Float[Array, "..."], x: Float[Array, "..."], c: float = 1.0) -> Float[Array, "..."]:
        """Project onto tangent space at x."""
        return self.manifold.tangent_proj(v, x, c, axis=self.hyperbolic_axis)


class HyperbolicActivation(nnx.Module):
    """
    Module to apply an activation function in the tangent space at the manifold origin.

    Parameters
    ----------
    manifold_module : module
        The hyperbolic manifold module (e.g., poincare, hyperboloid)
    activation : callable
        The activation function to apply in the tangent space at the manifold origin.
        Note: This is a static configuration.
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (default: -1).
        Note: This is a static configuration.
    backproject : bool
        Whether to project results back to the manifold (default: True).
        Note: This is a static configuration.

    Notes
    -----
    JIT Compatibility:
        This layer is fully compatible with nnx.jit.
    """

    def __init__(
        self,
        manifold_module: Any,
        activation: callable,
        hyperbolic_axis: int = -1,
        backproject: bool = True,
    ):
        # Static configuration (treated as compile-time constants for JIT)
        self.manifold = manifold_module
        self.hyperbolic_axis = hyperbolic_axis
        self.backproject = backproject
        self.activation = activation

    def __call__(self, x: Float[Array, "..."], c: float = 1.0) -> Float[Array, "..."]:
        """Apply activation in tangent space at origin."""
        # Map to tangent space at origin
        v = self.manifold.logmap_0(x, c, axis=self.hyperbolic_axis, backproject=self.backproject)
        # Apply activation
        v = self.activation(v)
        # Map back to manifold
        return self.manifold.expmap_0(v, c, axis=self.hyperbolic_axis, backproject=self.backproject)
