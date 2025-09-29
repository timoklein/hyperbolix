"""Base manifold interface and pure functional operations."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
import jax.numpy as jnp
from flax import struct

from ..config import RuntimeConfig


Array = jnp.ndarray


@runtime_checkable
class ManifoldOps(Protocol):
    """Protocol defining the pure functional interface for manifold operations.

    All operations are pure functions that can be JIT compiled with static arguments.
    The manifold struct provides these as bound methods with default static arguments.
    """

    def proj(
        self,
        x: Array,
        config: RuntimeConfig,
        axis: int = -1,
    ) -> Array:
        """Project point(s) onto the manifold."""
        ...

    def dist(
        self,
        x: Array,
        y: Array,
        config: RuntimeConfig,
        axis: int = -1,
        backproject: bool = True,
    ) -> Array:
        """Compute geodesic distance between points."""
        ...

    def dist_0(
        self,
        x: Array,
        config: RuntimeConfig,
        axis: int = -1,
    ) -> Array:
        """Compute geodesic distance from origin."""
        ...

    def expmap(
        self,
        v: Array,
        x: Array,
        config: RuntimeConfig,
        axis: int = -1,
        backproject: bool = True,
    ) -> Array:
        """Exponential map: tangent vector at x to manifold point."""
        ...

    def logmap(
        self,
        y: Array,
        x: Array,
        config: RuntimeConfig,
        axis: int = -1,
        backproject: bool = True,
    ) -> Array:
        """Logarithmic map: manifold point y to tangent vector at x."""
        ...

    def ptransp(
        self,
        v: Array,
        x: Array,
        y: Array,
        config: RuntimeConfig,
        axis: int = -1,
        backproject: bool = True,
    ) -> Array:
        """Parallel transport of tangent vector v from x to y."""
        ...

    def scalar_mul(
        self,
        r: Array,
        x: Array,
        config: RuntimeConfig,
        axis: int = -1,
        backproject: bool = True,
    ) -> Array:
        """Scalar multiplication on the manifold."""
        ...

    def tangent_proj(
        self,
        v: Array,
        x: Array,
        config: RuntimeConfig,
        axis: int = -1,
    ) -> Array:
        """Project vector v onto tangent space at x."""
        ...


@struct.dataclass
class ManifoldBase:
    """Base struct for all manifolds.

    This struct holds static configuration and provides bound methods
    that call the pure functional operations with default static arguments.
    """

    # Static configuration
    name: str
    config: RuntimeConfig

    # Default static arguments for JIT compilation
    default_axis: int = -1
    default_backproject: bool = True

    @property
    @abstractmethod
    def ops(self) -> ManifoldOps:
        """Get the pure functional operations for this manifold."""
        ...

    def proj(self, x: Array, *, axis: int = None) -> Array:
        """Project point(s) onto the manifold."""
        axis = axis if axis is not None else self.default_axis
        return self.ops.proj(x, config=self.config, axis=axis)

    def dist(
        self,
        x: Array,
        y: Array,
        *,
        axis: int = None,
        backproject: bool = None,
    ) -> Array:
        """Compute geodesic distance between points."""
        axis = axis if axis is not None else self.default_axis
        backproject = backproject if backproject is not None else self.default_backproject
        return self.ops.dist(x, y, config=self.config, axis=axis, backproject=backproject)

    def dist_0(self, x: Array, *, axis: int = None) -> Array:
        """Compute geodesic distance from origin."""
        axis = axis if axis is not None else self.default_axis
        return self.ops.dist_0(x, config=self.config, axis=axis)

    def expmap(
        self,
        v: Array,
        x: Array,
        *,
        axis: int = None,
        backproject: bool = None,
    ) -> Array:
        """Exponential map: tangent vector at x to manifold point."""
        axis = axis if axis is not None else self.default_axis
        backproject = backproject if backproject is not None else self.default_backproject
        return self.ops.expmap(v, x, config=self.config, axis=axis, backproject=backproject)

    def logmap(
        self,
        y: Array,
        x: Array,
        *,
        axis: int = None,
        backproject: bool = None,
    ) -> Array:
        """Logarithmic map: manifold point y to tangent vector at x."""
        axis = axis if axis is not None else self.default_axis
        backproject = backproject if backproject is not None else self.default_backproject
        return self.ops.logmap(y, x, config=self.config, axis=axis, backproject=backproject)

    def ptransp(
        self,
        v: Array,
        x: Array,
        y: Array,
        *,
        axis: int = None,
        backproject: bool = None,
    ) -> Array:
        """Parallel transport of tangent vector v from x to y."""
        axis = axis if axis is not None else self.default_axis
        backproject = backproject if backproject is not None else self.default_backproject
        return self.ops.ptransp(v, x, y, config=self.config, axis=axis, backproject=backproject)

    def scalar_mul(
        self,
        r: Array,
        x: Array,
        *,
        axis: int = None,
        backproject: bool = None,
    ) -> Array:
        """Scalar multiplication on the manifold."""
        axis = axis if axis is not None else self.default_axis
        backproject = backproject if backproject is not None else self.default_backproject
        return self.ops.scalar_mul(r, x, config=self.config, axis=axis, backproject=backproject)

    def tangent_proj(self, v: Array, x: Array, *, axis: int = None) -> Array:
        """Project vector v onto tangent space at x."""
        axis = axis if axis is not None else self.default_axis
        return self.ops.tangent_proj(v, x, config=self.config, axis=axis)

    def with_config(self, config: RuntimeConfig) -> 'ManifoldBase':
        """Create a new manifold with different runtime configuration."""
        return self.replace(config=config)

    def with_defaults(
        self,
        *,
        axis: int = None,
        backproject: bool = None,
    ) -> 'ManifoldBase':
        """Create a new manifold with different default static arguments."""
        updates = {}
        if axis is not None:
            updates['default_axis'] = axis
        if backproject is not None:
            updates['default_backproject'] = backproject
        return self.replace(**updates)