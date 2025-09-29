"""Euclidean manifold implementation using Flax struct.dataclass."""

from flax import struct
from typing import ClassVar

from ..config import RuntimeConfig, DEFAULT_CONFIG
from .base import ManifoldBase, ManifoldOps
from . import euclidean_ops


class EuclideanOps:
    """Pure functional operations for Euclidean manifold."""

    def proj(self, x, config, axis=-1):
        return euclidean_ops.proj(x, config, axis)

    def dist(self, x, y, config, axis=-1, backproject=True):
        return euclidean_ops.dist(x, y, config, axis, backproject)

    def dist_0(self, x, config, axis=-1):
        return euclidean_ops.dist_0(x, config, axis)

    def expmap(self, v, x, config, axis=-1, backproject=True):
        return euclidean_ops.expmap(v, x, config, axis, backproject)

    def logmap(self, y, x, config, axis=-1, backproject=True):
        return euclidean_ops.logmap(y, x, config, axis, backproject)

    def ptransp(self, v, x, y, config, axis=-1, backproject=True):
        return euclidean_ops.ptransp(v, x, y, config, axis, backproject)

    def scalar_mul(self, r, x, config, axis=-1, backproject=True):
        return euclidean_ops.scalar_mul(r, x, config, axis, backproject)

    def tangent_proj(self, v, x, config, axis=-1):
        return euclidean_ops.tangent_proj(v, x, config, axis)


@struct.dataclass
class Euclidean(ManifoldBase):
    """Euclidean manifold.

    Represents standard Euclidean space R^n with the usual inner product.
    All operations are trivial (identity, addition, subtraction).
    """

    # Static configuration
    name: str = "Euclidean"
    config: RuntimeConfig = DEFAULT_CONFIG

    # Default static arguments for JIT compilation
    default_axis: int = -1
    default_backproject: bool = True

    # Operations instance (class variable to avoid recompilation)
    _ops: ClassVar[EuclideanOps] = EuclideanOps()

    @property
    def ops(self) -> ManifoldOps:
        """Get the pure functional operations for this manifold."""
        return self._ops


def create_euclidean(
    config: RuntimeConfig = None,
    *,
    dtype: str = None,
    axis: int = -1,
    backproject: bool = True,
) -> Euclidean:
    """Create a Euclidean manifold with specified configuration.

    Args:
        config: Runtime configuration. If None, uses default for dtype.
        dtype: Data type ('float32' or 'float64'). Ignored if config provided.
        axis: Default axis for operations.
        backproject: Default backproject setting.

    Returns:
        Euclidean manifold instance.
    """
    if config is None:
        if dtype is not None:
            from ..config import get_dtype_config
            config = get_dtype_config(dtype)
        else:
            config = DEFAULT_CONFIG

    return Euclidean(
        config=config,
        default_axis=axis,
        default_backproject=backproject,
    )
