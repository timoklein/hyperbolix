"""Hyperbolic regularization modules using HRC from Hypformer.

This module contains NNX regularization layers (Dropout, LayerNorm, BatchNorm) that
use the Hyperbolic Regularization Component (HRC) pattern with curvature-change support.

For the core HRC function, see hyperboloid_core module.
For activation functions using HRC, see hyperboloid_activations module.
For the Hyperbolic Transformation Component (HTC), see hyperboloid_linear module.

Key components:
- HRCDropout: Hyperbolic dropout with curvature change
- HRCLayerNorm: Hyperbolic layer normalization with curvature change
- HRCBatchNorm: Hyperbolic batch normalization with curvature change

References
----------
Hypformer paper (citation to be added)
"""

from flax import nnx
from jaxtyping import Array, Float

from .hyperboloid_core import hrc

# Flax NNX Modules


class HRCDropout(nnx.Module):
    """Hyperbolic Regularization Component with dropout.

    Applies dropout to spatial components of hyperboloid points, then reconstructs
    the time component for the output curvature.

    Parameters
    ----------
    rate : float
        Dropout probability (fraction of units to drop).
    rngs : nnx.Rngs
        Random number generators for dropout.
    eps : float, optional
        Small value for numerical stability (default: 1e-7).

    Attributes
    ----------
    dropout : nnx.Dropout
        Flax dropout layer.
    eps : float
        Numerical stability parameter.

    Examples
    --------
    >>> from flax import nnx
    >>> from hyperbolix.nn_layers import HRCDropout
    >>>
    >>> dropout = HRCDropout(rate=0.1, rngs=nnx.Rngs(dropout=42))
    >>> y = dropout(x, c_in=1.0, c_out=1.0, deterministic=False)
    """

    def __init__(self, rate: float, *, rngs: nnx.Rngs, eps: float = 1e-7):
        self.dropout = nnx.Dropout(rate, rngs=rngs)
        self.eps = eps

    def __call__(
        self,
        x: Float[Array, "batch dim_plus_1"],
        c_in: float = 1.0,
        c_out: float = 1.0,
        deterministic: bool = False,
    ) -> Float[Array, "batch dim_plus_1"]:
        """Apply HRC dropout.

        Parameters
        ----------
        x : Array of shape (batch, dim+1)
            Input points on hyperboloid with curvature c_in.
        c_in : float, optional
            Input curvature (default: 1.0).
        c_out : float, optional
            Output curvature (default: 1.0).
        deterministic : bool, optional
            If True, no dropout is applied (for evaluation mode).

        Returns
        -------
        y : Array of shape (batch, dim+1)
            Output points on hyperboloid with curvature c_out.
        """

        def drop_fn(z):
            return self.dropout(z, deterministic=deterministic)

        return hrc(x, drop_fn, c_in, c_out, self.eps)


class HRCLayerNorm(nnx.Module):
    """Hyperbolic Regularization Component with layer normalization.

    Applies layer normalization to spatial components of hyperboloid points,
    then reconstructs the time component for the output curvature.

    Parameters
    ----------
    num_features : int
        Number of spatial features to normalize.
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    epsilon : float, optional
        Small value for numerical stability in layer norm (default: 1e-5).
    eps : float, optional
        Small value for numerical stability in HRC (default: 1e-7).

    Attributes
    ----------
    ln : nnx.LayerNorm
        Flax layer normalization.
    eps : float
        Numerical stability parameter for HRC.

    Examples
    --------
    >>> from flax import nnx
    >>> from hyperbolix.nn_layers import HRCLayerNorm
    >>>
    >>> ln = HRCLayerNorm(num_features=64, rngs=nnx.Rngs(0))
    >>> y = ln(x, c_in=1.0, c_out=2.0)
    """

    def __init__(
        self,
        num_features: int,
        *,
        rngs: nnx.Rngs,
        epsilon: float = 1e-5,
        eps: float = 1e-7,
    ):
        self.ln = nnx.LayerNorm(num_features, epsilon=epsilon, rngs=rngs)
        self.eps = eps

    def __call__(
        self,
        x: Float[Array, "batch dim_plus_1"],
        c_in: float = 1.0,
        c_out: float = 1.0,
    ) -> Float[Array, "batch dim_plus_1"]:
        """Apply HRC layer normalization.

        Parameters
        ----------
        x : Array of shape (batch, dim+1)
            Input points on hyperboloid with curvature c_in.
        c_in : float, optional
            Input curvature (default: 1.0).
        c_out : float, optional
            Output curvature (default: 1.0).

        Returns
        -------
        y : Array of shape (batch, dim+1)
            Output points on hyperboloid with curvature c_out.
        """
        return hrc(x, self.ln, c_in, c_out, self.eps)


class HRCBatchNorm(nnx.Module):
    """Hyperbolic Regularization Component with batch normalization.

    Applies batch normalization to spatial components of hyperboloid points,
    then reconstructs the time component for the output curvature.

    Parameters
    ----------
    num_features : int
        Number of spatial features to normalize.
    rngs : nnx.Rngs
        Random number generators for parameter initialization.
    momentum : float, optional
        Momentum for running statistics (default: 0.99).
    epsilon : float, optional
        Small value for numerical stability in batch norm (default: 1e-5).
    eps : float, optional
        Small value for numerical stability in HRC (default: 1e-7).

    Attributes
    ----------
    bn : nnx.BatchNorm
        Flax batch normalization.
    eps : float
        Numerical stability parameter for HRC.

    Notes
    -----
    Training vs Evaluation Mode:
        During training (use_running_average=False), batch norm computes statistics
        from the current batch and updates running averages. During evaluation
        (use_running_average=True), it uses the accumulated running statistics.

    Examples
    --------
    >>> from flax import nnx
    >>> from hyperbolix.nn_layers import HRCBatchNorm
    >>>
    >>> # Create batch norm layer
    >>> bn = HRCBatchNorm(num_features=64, rngs=nnx.Rngs(0))
    >>>
    >>> # Training mode
    >>> y_train = bn(x, c_in=1.0, c_out=2.0, use_running_average=False)
    >>>
    >>> # Evaluation mode
    >>> y_eval = bn(x, c_in=1.0, c_out=2.0, use_running_average=True)
    """

    def __init__(
        self,
        num_features: int,
        *,
        rngs: nnx.Rngs,
        momentum: float = 0.99,
        epsilon: float = 1e-5,
        eps: float = 1e-7,
    ):
        self.bn = nnx.BatchNorm(
            num_features,
            momentum=momentum,
            epsilon=epsilon,
            rngs=rngs,
        )
        self.eps = eps

    def __call__(
        self,
        x: Float[Array, "batch dim_plus_1"],
        c_in: float = 1.0,
        c_out: float = 1.0,
        use_running_average: bool | None = None,
    ) -> Float[Array, "batch dim_plus_1"]:
        """Apply HRC batch normalization.

        Parameters
        ----------
        x : Array of shape (batch, dim+1)
            Input points on hyperboloid with curvature c_in.
        c_in : float, optional
            Input curvature (default: 1.0).
        c_out : float, optional
            Output curvature (default: 1.0).
        use_running_average : bool or None, optional
            If True, use running statistics (eval mode).
            If False, use batch statistics (train mode).
            If None, use the default set during initialization.

        Returns
        -------
        y : Array of shape (batch, dim+1)
            Output points on hyperboloid with curvature c_out.
        """

        def bn_fn(z):
            return self.bn(z, use_running_average=use_running_average)

        return hrc(x, bn_fn, c_in, c_out, self.eps)
