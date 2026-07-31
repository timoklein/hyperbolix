"""Tests for Hyperboloid convolutional layers and HCat operation.

The shared forward / on-manifold / JIT / gradient / tangent-input contract for
every layer in the library lives in ``test_layer_contract.py``; only
HypConv2DHyperboloid and hcat-specific tests stay here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from scipy.special import digamma as scipy_digamma

from hyperbolix.manifolds import Hyperboloid
from hyperbolix.nn_layers.hyperboloid_conv import HypConv2DHyperboloid
from hyperbolix.nn_layers.hyperboloid_core import hcat_ambient_dim

# ============================================================================
# HCat Operation Tests
# ============================================================================


@pytest.mark.parametrize("N,n,c", [(2, 3, 1.0), (3, 4, 1.0), (4, 5, 0.5), (1, 3, 1.0)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hcat_output_on_manifold(N, n, c, dtype):
    """Test that HCat output lies on the Hyperboloid manifold."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)

    # Create N points on n-dimensional hyperboloid
    points = []
    for _i in range(N):
        key, subkey = jax.random.split(key)
        tangent = jax.random.normal(subkey, (n,), dtype=dtype) * 0.1
        tangent = tangent.at[0].set(0)  # Set time coordinate to 0 (tangent at origin)
        point = manifold.expmap_0(tangent, c)
        points.append(point)

    points = jnp.stack(points)  # (N, n)

    # Apply HCat
    result = manifold.hcat(points, c)

    # Check output dimension
    # Input: N points of ambient dimension n (manifold dim d = n-1)
    # Output: ambient dimension dN + 1 = (n-1)*N + 1
    d = n - 1  # Manifold dimension
    expected_dim = d * N + 1
    assert result.shape == (expected_dim,), f"Expected shape ({expected_dim},), got {result.shape}"

    # Check output is on manifold
    # For a (dN)-dimensional manifold with curvature c:
    # Lorentz constraint: -x[0]^2 + sum(x[1:]^2) = -1/c
    lorentz_product = -(result[0] ** 2) + jnp.sum(result[1:] ** 2)
    expected = -1.0 / c
    tolerance = 1e-5 if dtype == jnp.float32 else 1e-10
    assert jnp.abs(lorentz_product - expected) < tolerance, f"HCat output not on manifold: {lorentz_product} != {expected}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hcat_single_point(dtype):
    """Test HCat with a single point (edge case N=1)."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    n, c = 5, 1.0

    # Create single point
    tangent = jax.random.normal(key, (n,), dtype=dtype) * 0.1
    tangent = tangent.at[0].set(0)
    point = manifold.expmap_0(tangent, c)
    points = point.reshape(1, n)  # (1, n)

    # Apply HCat
    result = manifold.hcat(points, c)

    # For N=1, time coordinate should be sqrt(x[0]^2 + 0) = |x[0]|
    # and space coordinates should just be x[1:]
    assert result.shape == (n,)
    assert jnp.abs(result[0] - jnp.abs(point[0])) < 1e-6
    assert jnp.allclose(result[1:], point[1:], atol=1e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hcat_dimensionality(dtype):
    """Test HCat correctly increases dimensionality."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    test_cases = [(2, 3), (3, 4), (5, 6)]

    for N, n in test_cases:
        # Create N points on n-dimensional hyperboloid (n is ambient dim)
        points = []
        for _i in range(N):
            key, subkey = jax.random.split(key)
            tangent = jax.random.normal(subkey, (n,), dtype=dtype) * 0.1
            tangent = tangent.at[0].set(0)
            point = manifold.expmap_0(tangent, 1.0)
            points.append(point)

        points = jnp.stack(points)  # (N, n)

        # Apply HCat
        result = manifold.hcat(points, 1.0)

        # Check dimensionality: (n-1)*N + 1
        d = n - 1  # Manifold dimension
        expected_dim = d * N + 1
        assert result.shape == (expected_dim,), f"Expected ({expected_dim},), got {result.shape}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hcat_time_coordinate_formula(dtype):
    """Test HCat time coordinate computation formula."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    N, n, c = 3, 4, 1.0

    # Create N points
    points = []
    for _i in range(N):
        key, subkey = jax.random.split(key)
        tangent = jax.random.normal(subkey, (n,), dtype=dtype) * 0.1
        tangent = tangent.at[0].set(0)
        point = manifold.expmap_0(tangent, c)
        points.append(point)

    points = jnp.stack(points)  # (N, n)

    # Apply HCat
    result = manifold.hcat(points, c)

    # Manually compute expected time coordinate using the CORRECT formula
    # Formula: sqrt(sum(x_i[0]^2) - (N-1)/c)  [note the MINUS]
    time_coords = points[:, 0]  # (N,)
    expected_time = jnp.sqrt(jnp.sum(time_coords**2) - (N - 1) / c)

    tolerance = 1e-5 if dtype == jnp.float32 else 1e-10
    assert jnp.abs(result[0] - expected_time) < tolerance, f"Time coordinate mismatch: {result[0]} != {expected_time}"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hcat_space_concatenation(dtype):
    """Test HCat correctly concatenates space coordinates."""
    key = jax.random.PRNGKey(42)
    manifold = Hyperboloid(dtype=dtype)
    N, n, c = 3, 4, 1.0

    # Create N points
    points = []
    for _i in range(N):
        key, subkey = jax.random.split(key)
        tangent = jax.random.normal(subkey, (n,), dtype=dtype) * 0.1
        tangent = tangent.at[0].set(0)
        point = manifold.expmap_0(tangent, c)
        points.append(point)

    points = jnp.stack(points)  # (N, n)

    # Apply HCat
    result = manifold.hcat(points, c)

    # Check space coordinates are correctly concatenated
    expected_space = points[:, 1:].reshape(-1)  # (N*(n-1),)
    actual_space = result[1:]  # (N*n - 1,)

    assert jnp.allclose(actual_space, expected_space, atol=1e-6)


# ============================================================================
# Log-Radius Concatenation Tests (Shi et al. 2026, Sec. 4.3)
# ============================================================================


def _random_hyperboloid_points(key, N, n, c, dtype, scale=0.1):
    """Create N points on the (n-1)-dim hyperboloid (ambient dim n) via expmap_0."""
    manifold = Hyperboloid(dtype=dtype)
    points = []
    for _i in range(N):
        key, subkey = jax.random.split(key)
        tangent = jax.random.normal(subkey, (n,), dtype=dtype) * scale
        tangent = tangent.at[0].set(0)  # tangent at origin (time component 0)
        points.append(manifold.expmap_0(tangent, c))
    return jnp.stack(points)  # (N, n)


def _expected_log_radius(k):
    """E[log ||v||] for v ~ N(0, I_k), from the chi distribution: 0.5*(psi(k/2) + log 2).

    Independent oracle for the log_radius_concat contract — derived from the chi^2
    moment-generating identity, not transcribed from the library's digamma call.
    """
    return 0.5 * (float(scipy_digamma(k / 2.0)) + float(np.log(2.0)))


def _implied_scale(points_Nn, result_A):
    """Recover the scalar block scale the library applied, from its largest spatial entry."""
    ref_flat = np.asarray(points_Nn, dtype=np.float64)[:, 1:].reshape(-1)
    idx = int(np.argmax(np.abs(ref_flat)))
    return float(np.asarray(result_A, dtype=np.float64)[1:][idx] / ref_flat[idx])


@pytest.mark.parametrize("N,n,c", [(2, 3, 1.0), (3, 4, 1.0), (4, 5, 0.5), (1, 3, 1.0)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_log_radius_concat_output_on_manifold(N, n, c, dtype):
    """Log-radius concat output lies on the (dN)-dimensional Hyperboloid."""
    manifold = Hyperboloid(dtype=dtype)
    points = _random_hyperboloid_points(jax.random.PRNGKey(42), N, n, c, dtype)

    result = manifold.log_radius_concat(points, c)

    d = n - 1
    expected_dim = d * N + 1
    assert result.shape == (expected_dim,)

    lorentz_product = -(result[0] ** 2) + jnp.sum(result[1:] ** 2)
    tolerance = 1e-5 if dtype == jnp.float32 else 1e-10
    assert jnp.abs(lorentz_product - (-1.0 / c)) < tolerance


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_log_radius_concat_single_point_identity(dtype):
    """For N=1 there is nothing to rescale, so log_radius_concat reduces to hcat (identity)."""
    manifold = Hyperboloid(dtype=dtype)
    n, c = 5, 1.0
    points = _random_hyperboloid_points(jax.random.PRNGKey(42), 1, n, c, dtype)

    result = manifold.log_radius_concat(points, c)

    assert result.shape == (n,)
    # Scale == 1 ⇒ returns the input point and matches the unscaled hcat exactly.
    assert jnp.allclose(result, points[0], atol=1e-6)
    assert jnp.allclose(result, manifold.hcat(points, c), atol=1e-6)
    assert abs(_implied_scale(points, result) - 1.0) < 1e-6


@pytest.mark.parametrize("N,d,c", [(9, 32, 1.0), (9, 3, 1.0), (4, 16, 0.3), (2, 8, 1.0), (3, 5, 2.0)])
def test_log_radius_concat_preserves_expected_log_radius(N, d, c):
    """LogCat's declared contract: E[log‖v_spatial‖] does not move with the block count N.

    Independent oracle — for Gaussian spatial parts ``‖v‖² ~ χ²_k`` so
    ``E[log‖v‖] = ½(ψ(k/2) + log 2)`` (:func:`_expected_log_radius`), computed with
    SciPy and never read from the library. Discriminating cases, all rejected here by
    margins ≥ 0.34 against a 0.02 tolerance: scale 1 (plain ``hcat``, off by
    ``+½·log N``), scale 0 (collapse, ``-inf``), and the inverted digamma difference
    that shipped through hyperbolix 1.0.0 (off by ``+log N`` — *worse* than hcat).
    """
    n_samples = 20_000
    rng = np.random.default_rng(20260731)
    space_SNd = rng.standard_normal((n_samples, N, d))
    time_SN = np.sqrt(1.0 / c + np.sum(space_SNd**2, axis=-1))
    points_SNn = jnp.asarray(np.concatenate([time_SN[..., None], space_SNd], axis=-1), dtype=jnp.float64)

    manifold = Hyperboloid(dtype=jnp.float64)
    out_SA = jax.vmap(manifold.log_radius_concat, in_axes=(0, None))(points_SNn, c)

    measured = float(np.mean(np.log(np.linalg.norm(np.asarray(out_SA)[:, 1:], axis=-1))))
    target = _expected_log_radius(d)  # per-block radius, the quantity that must be preserved
    assert abs(measured - target) < 0.02, f"E[log r] moved: {measured:.4f} vs target {target:.4f}"

    # Not vacuous: plain concatenation (scale 1) sits at the *widened* chi radius instead,
    # which is outside the tolerance above — so scale=1 and scale=0 both fail this test.
    hcat_out_SA = jax.vmap(manifold.hcat, in_axes=(0, None))(points_SNn, c)
    hcat_measured = float(np.mean(np.log(np.linalg.norm(np.asarray(hcat_out_SA)[:, 1:], axis=-1))))
    assert abs(hcat_measured - _expected_log_radius(N * d)) < 0.02  # sampler/oracle agree
    assert abs(hcat_measured - target) > 0.02


@pytest.mark.parametrize("N,n,c", [(2, 3, 1.0), (3, 4, 0.5), (4, 5, 1.0)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_log_radius_concat_blocks_share_one_shrinking_scale(N, n, c, dtype):
    """Every block is scaled by the *same* scalar, blocks stay in input order, and N>1 shrinks.

    Structural companion to the statistical invariant test: pins the block layout
    (a per-block or per-coordinate scale would fail) without transcribing the digamma
    formula. Widening the spatial dimension must shrink each block, so ``scale < 1``.
    """
    manifold = Hyperboloid(dtype=dtype)
    points = _random_hyperboloid_points(jax.random.PRNGKey(7), N, n, c, dtype)

    result = manifold.log_radius_concat(points, c)

    scale = _implied_scale(points, result)
    expected_space = (scale * np.asarray(points[:, 1:], dtype=np.float64)).reshape(-1)
    tolerance = 1e-5 if dtype == jnp.float32 else 1e-10
    assert np.allclose(np.asarray(result[1:], dtype=np.float64), expected_space, atol=tolerance)
    assert scale < 1.0


@pytest.mark.parametrize("N,n,c", [(2, 3, 1.0), (3, 4, 0.5), (4, 5, 1.0)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_log_radius_concat_time_formula(N, n, c, dtype):
    """Time coordinate equals sqrt(1/c + s^2 * sum(t_i^2 - 1/c)) for the applied scale s."""
    manifold = Hyperboloid(dtype=dtype)
    points = _random_hyperboloid_points(jax.random.PRNGKey(11), N, n, c, dtype)

    result = manifold.log_radius_concat(points, c)

    scale = _implied_scale(points, result)
    time_coords = np.asarray(points[:, 0], dtype=np.float64)
    expected_time = np.sqrt(1.0 / c + scale**2 * np.sum(time_coords**2 - 1.0 / c))
    tolerance = 1e-5 if dtype == jnp.float32 else 1e-10
    assert abs(float(result[0]) - expected_time) < tolerance


# ============================================================================
# HypConv2DHyperboloid Layer Tests
# ============================================================================


def test_hypconv_hyperboloid_kernel_init_reset_params():
    """Default kernel init is U(-0.02, 0.02); reset_params='fan_in' gives
    std=sqrt(1/hcat_out_ambient_dim)."""
    manifold = Hyperboloid(dtype=jnp.float32)
    in_channels, out_channels, kernel_size = 9, 17, 3

    layer_default = HypConv2DHyperboloid(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        rngs=nnx.Rngs(42),
    )
    layer_fan_in = HypConv2DHyperboloid(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        rngs=nnx.Rngs(42),
        reset_params="fan_in",
    )

    bound = 0.02
    expected_default_std = bound / jnp.sqrt(3.0)  # std of U(-bound, bound)
    assert abs(float(jnp.std(layer_default.kernel[...])) - float(expected_default_std)) < 0.2 * float(expected_default_std)

    hcat_out_ambient_dim = hcat_ambient_dim(in_channels, (kernel_size, kernel_size))
    expected_fan_in_std = float(jnp.sqrt(1.0 / hcat_out_ambient_dim))
    assert abs(float(jnp.std(layer_fan_in.kernel[...])) - expected_fan_in_std) < 0.2 * expected_fan_in_std


@pytest.mark.parametrize("c", [0.5, 1.0])
def test_hypconv_hyperboloid_1x1_matches_numpy_fhcnn(c):
    """A 1x1 conv is HCat(single point) = identity followed by the FHCNN linear map.

    Value oracle (audit A6-02): the whole pipeline is transcribed in NumPy from
    Bdeir et al. 2023 --- ``z = W x + b``, spatial output ``z_s`` kept as is, time
    coordinate reconstructed as ``sqrt(||z_s||^2 + 1/c)``. A collapsed or negated
    spatial branch, or a mis-wired patch reshape, fails here.
    """
    dtype = jnp.float64
    manifold = Hyperboloid(dtype=dtype)
    batch, height, width, in_channels, out_channels = 2, 3, 3, 4, 5

    v = jax.random.normal(jax.random.PRNGKey(0), (batch, height, width, in_channels), dtype=dtype) * 0.3
    v = v.at[..., 0].set(0.0)
    x = jax.vmap(jax.vmap(jax.vmap(lambda p: manifold.expmap_0(p, c))))(v)

    layer = HypConv2DHyperboloid(
        manifold_module=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        padding="VALID",
        rngs=nnx.Rngs(0),
        param_dtype=dtype,
    )
    layer.kernel[...] = jax.random.normal(jax.random.PRNGKey(1), (out_channels, in_channels), dtype=dtype) * 0.4
    layer.bias[...] = jax.random.normal(jax.random.PRNGKey(2), (1, out_channels), dtype=dtype) * 0.2

    y = layer(x, c=c)

    x_NC = np.asarray(x, dtype=np.float64).reshape(-1, in_channels)
    z_NO = x_NC @ np.asarray(layer.kernel[...], dtype=np.float64).T + np.asarray(layer.bias[...], dtype=np.float64)
    ys_ND = z_NO[:, 1:]
    yt_N1 = np.sqrt(np.sum(ys_ND**2, axis=-1, keepdims=True) + 1.0 / c)
    expected = np.concatenate([yt_N1, ys_ND], axis=-1).reshape(batch, height, width, out_channels)

    assert np.allclose(np.asarray(y), expected, atol=1e-12)
    assert np.max(np.abs(expected[..., 1:])) > 0.05  # oracle is non-degenerate


def test_hypconv_hyperboloid_invalid_reset_params_errors():
    with pytest.raises(ValueError, match="reset_params"):
        HypConv2DHyperboloid(
            manifold_module=Hyperboloid(dtype=jnp.float32),
            in_channels=9,
            out_channels=17,
            kernel_size=3,
            rngs=nnx.Rngs(0),
            reset_params="bogus",
        )
