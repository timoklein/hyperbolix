"""Probe 2: per-layer spatial-norm profile of a depth-3 ILNN conv stack at init.

Dimension key:
  B: batch   H,W: spatial   C: ambient channels (= spatial + 1)
Compares (buggy LogCat sign, std=0.02) / (fixed sign, std=0.02) /
(fixed sign, candidate norm-preserving stds).
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.scipy.special import digamma

jax.config.update("jax_enable_x64", True)

import hyperbolix.manifolds.hyperboloid as H  # noqa: E402
from hyperbolix.manifolds import Hyperboloid  # noqa: E402
from hyperbolix.nn_layers import HypConv2DHyperboloidILNN  # noqa: E402

FIXED = H._log_radius_concat


def _buggy_log_radius_concat(points, c=1.0):
    N, ambient = points.shape
    d = ambient - 1
    n_total = N * d
    scale = jnp.exp(0.5 * (digamma(n_total / 2.0) - digamma(d / 2.0)))  # SHIPPED (wrong) direction
    time_N, space_ND = points[:, 0], points[:, 1:]
    space = (scale * space_ND).reshape(-1)
    t2 = 1.0 / c + scale**2 * jnp.sum(time_N**2 - 1.0 / c)
    return jnp.concatenate([jnp.sqrt(jnp.maximum(t2, H.MIN_NORM))[None], space])


def run(std, buggy, *, depth=3, c=0.1, hw=8, channels=33, kernel=3, seed=0, verbose=True):
    H._log_radius_concat = _buggy_log_radius_concat if buggy else FIXED
    manifold = Hyperboloid(dtype=jnp.float64)
    key = jax.random.PRNGKey(seed)
    tangent_BHWC = jax.random.normal(key, (2, hw, hw, channels), dtype=jnp.float64) * 0.3
    tangent_BHWC = tangent_BHWC.at[..., 0].set(0.0)
    x = jax.vmap(jax.vmap(jax.vmap(lambda t: manifold.expmap_0(t, c))))(tangent_BHWC)

    norms = [float(jnp.mean(jnp.linalg.norm(x[..., 1:], axis=-1)))]
    interior = [float(jnp.mean(jnp.linalg.norm(x[:, 1:-1, 1:-1, 1:], axis=-1)))]
    rngs = nnx.Rngs(1234)
    for _ in range(depth):
        layer = HypConv2DHyperboloidILNN(
            manifold_module=manifold,
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel,
            rngs=rngs,
            kernel_init_std=std,
            param_dtype=jnp.float64,
        )
        x = layer(x, c=c)
        norms.append(float(jnp.mean(jnp.linalg.norm(x[..., 1:], axis=-1))))
        interior.append(float(jnp.mean(jnp.linalg.norm(x[:, 1:-1, 1:-1, 1:], axis=-1))))
    H._log_radius_concat = FIXED
    ratios = [norms[i + 1] / norms[i] for i in range(depth)]
    if verbose:
        tag = "buggy" if buggy else "fixed"
        print(
            f"  [{tag}] std={std:<8.5f} norms={[f'{v:.4g}' for v in norms]} "
            f"ratios={[f'{v:.3g}' for v in ratios]} interior={[f'{v:.4g}' for v in interior]}"
        )
    return norms, ratios


def kappa(N, d):
    """RMS amplification of the FIXED LogCat: s * sqrt(N)."""
    return float(np.sqrt(N) * np.exp(0.5 * (float(digamma(d / 2)) - float(digamma(N * d / 2)))))


for channels, kernel in [(33, 3), (17, 3), (9, 3), (65, 3), (17, 2)]:
    out_spatial = channels - 1
    N = kernel * kernel
    k = kappa(N, channels - 1)
    print(f"\nchannels={channels} kernel={kernel}  out_spatial={out_spatial}  N={N}  kappa={k:.4f}")
    print(f"  candidates: 1/sqrt(O)={1 / np.sqrt(out_spatial):.5f}  1/(sqrt(O)*kappa)={1 / (np.sqrt(out_spatial) * k):.5f}")
    run(0.02, buggy=True, channels=channels, kernel=kernel)
    run(0.02, buggy=False, channels=channels, kernel=kernel)
    run(1 / np.sqrt(out_spatial), buggy=False, channels=channels, kernel=kernel)
    run(1 / (np.sqrt(out_spatial) * k), buggy=False, channels=channels, kernel=kernel)
