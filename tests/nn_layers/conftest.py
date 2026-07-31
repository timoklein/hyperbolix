"""Shared input factories for the ``tests/nn_layers`` suite.

Only the two Poincaré input builders that were duplicated verbatim across
``test_poincare_batchnorm.py`` and ``test_poincare_conv.py`` live here. They are
exposed as fixtures returning the callable (rather than module-level functions)
so nothing about collection or parametrization of the other files changes.

Dimension key:
  N: flattened batch     C: channels (ball dim)
"""

import jax
import pytest

from hyperbolix.manifolds import Poincare


def _make_poincare_points(key, shape, c, dtype):
    """Random points on the Poincaré ball via expmap_0 of small tangent vectors."""
    manifold = Poincare(dtype=dtype)
    tangent = jax.random.normal(key, shape, dtype=dtype) * 0.1
    if tangent.ndim == 1:
        return manifold.expmap_0(tangent, c)
    flat = tangent.reshape(-1, shape[-1])
    pts = jax.vmap(manifold.expmap_0, in_axes=(0, None))(flat, c)
    return pts.reshape(shape)


def _make_tangent_input(key, shape, dtype):
    """Random tangent-space input (small vectors near the origin)."""
    return jax.random.normal(key, shape, dtype=dtype) * 0.1


@pytest.fixture
def make_poincare_points():
    """Factory: ``(key, shape, c, dtype) -> on-ball points of that shape``."""
    return _make_poincare_points


@pytest.fixture
def make_tangent_input():
    """Factory: ``(key, shape, dtype) -> small tangent vectors of that shape``."""
    return _make_tangent_input
