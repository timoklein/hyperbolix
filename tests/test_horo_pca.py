"""Tests for JAX HoroPCA implementation."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hyperbolix.manifolds import hyperboloid
from hyperbolix.utils.horo_pca import HoroPCA, center_data, compute_frechet_mean


class TestComputeFrechetMean:
    """Tests for Fréchet mean computation."""

    def test_frechet_mean_shape(self):
        """Test that Fréchet mean has correct shape."""
        key = jax.random.PRNGKey(42)
        n_points = 50
        dim = 10
        c = 1.0

        # Generate random hyperboloid points
        x = jax.random.normal(key, (n_points, dim + 1))
        x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, c)

        # Compute Fréchet mean
        mean = compute_frechet_mean(x, c, max_iters=100)

        assert mean.shape == (1, dim + 1), f"Expected shape (1, {dim + 1}), got {mean.shape}"

    def test_frechet_mean_on_manifold(self):
        """Test that Fréchet mean lies on the hyperboloid."""
        key = jax.random.PRNGKey(42)
        n_points = 50
        dim = 10
        c = 1.0

        # Generate random hyperboloid points
        x = jax.random.normal(key, (n_points, dim + 1))
        x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, c)

        # Compute Fréchet mean
        mean = compute_frechet_mean(x, c, max_iters=100)

        # Check if mean is on manifold
        is_valid = hyperboloid.is_in_manifold(mean[0], c, atol=1e-4)
        assert is_valid, "Fréchet mean should lie on the hyperboloid"

    def test_frechet_mean_single_point(self):
        """Test that Fréchet mean of a single point is the point itself."""
        key = jax.random.PRNGKey(42)
        dim = 10
        c = 1.0

        # Generate single hyperboloid point
        x = jax.random.normal(key, (1, dim + 1))
        x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, c)

        # Compute Fréchet mean
        mean = compute_frechet_mean(x, c, max_iters=100)

        # Mean should be close to the single point
        distance = hyperboloid.dist(x[0], mean[0], c, hyperboloid.VERSION_DEFAULT)
        assert distance < 1e-3, f"Distance between mean and single point should be small, got {distance}"

    def test_frechet_mean_origin_cluster(self):
        """Test Fréchet mean of points clustered near origin."""
        key = jax.random.PRNGKey(42)
        n_points = 50
        dim = 10
        c = 1.0

        # Generate points near origin
        origin = jnp.zeros(dim + 1)
        origin = origin.at[0].set(1.0 / jnp.sqrt(c))
        origin = hyperboloid.proj(origin, c)

        # Add small perturbations
        x = jax.random.normal(key, (n_points, dim + 1)) * 0.1
        x = jax.vmap(lambda p: hyperboloid.proj(origin + p, c))(x)

        # Compute Fréchet mean
        mean = compute_frechet_mean(x, c, max_iters=100)

        # Mean should be close to origin
        dist_to_origin = hyperboloid.dist_0(mean[0], c, hyperboloid.VERSION_DEFAULT)
        assert dist_to_origin < 1.0, f"Mean should be close to origin, distance: {dist_to_origin}"


class TestCenterData:
    """Tests for data centering."""

    def test_center_data_shape(self):
        """Test that centered data has correct shape."""
        key = jax.random.PRNGKey(42)
        n_points = 50
        dim = 10
        c = 1.0

        # Generate random hyperboloid points
        x = jax.random.normal(key, (n_points, dim + 1))
        x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, c)

        # Compute mean
        mean = compute_frechet_mean(x, c, max_iters=100)

        # Center data
        x_centered = center_data(x, mean, c)

        assert x_centered.shape == x.shape, f"Expected shape {x.shape}, got {x_centered.shape}"

    def test_center_data_on_manifold(self):
        """Test that centered points remain on hyperboloid."""
        key = jax.random.PRNGKey(42)
        n_points = 50
        dim = 10
        c = 1.0

        # Generate random hyperboloid points
        x = jax.random.normal(key, (n_points, dim + 1))
        x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, c)

        # Compute mean and center
        mean = compute_frechet_mean(x, c, max_iters=100)
        x_centered = center_data(x, mean, c)

        # Check if all points are on manifold
        is_valid = jax.vmap(hyperboloid.is_in_manifold, in_axes=(0, None, None))(x_centered, c, 1e-4)
        assert jnp.all(is_valid), "All centered points should lie on the hyperboloid"

    def test_center_data_mean_at_origin(self):
        """Test that centering moves mean close to origin."""
        key = jax.random.PRNGKey(42)
        n_points = 50
        dim = 10
        c = 1.0

        # Generate random hyperboloid points
        x = jax.random.normal(key, (n_points, dim + 1))
        x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, c)

        # Compute mean and center
        mean = compute_frechet_mean(x, c, max_iters=100)
        x_centered = center_data(x, mean, c)

        # Compute new mean of centered data
        new_mean = compute_frechet_mean(x_centered, c, max_iters=100)

        # New mean should be close to origin
        dist_to_origin = hyperboloid.dist_0(new_mean[0], c, hyperboloid.VERSION_DEFAULT)
        assert dist_to_origin < 0.5, f"Centered data mean should be near origin, distance: {dist_to_origin}"


class TestHoroPCA:
    """Tests for HoroPCA class."""

    def test_initialization_hyperboloid(self):
        """Test HoroPCA initialization with hyperboloid manifold."""
        n_components = 5
        n_in_features = 11
        c = 1.0
        rngs = nnx.Rngs(42)

        model = HoroPCA(
            n_components=n_components,
            n_in_features=n_in_features,
            manifold_name="hyperboloid",
            c=c,
            rngs=rngs,
        )

        assert model.n_components == n_components
        assert model.n_in_features == n_in_features
        assert model.manifold_name == "hyperboloid"
        assert model.Q[...].shape == (n_components, n_in_features - 1)

    def test_initialization_poincare(self):
        """Test HoroPCA initialization with Poincaré manifold."""
        n_components = 5
        n_in_features = 10
        c = 1.0
        rngs = nnx.Rngs(42)

        model = HoroPCA(
            n_components=n_components,
            n_in_features=n_in_features,
            manifold_name="poincare",
            c=c,
            rngs=rngs,
        )

        assert model.n_components == n_components
        assert model.n_in_features == n_in_features
        assert model.manifold_name == "poincare"
        assert model.Q[...].shape == (n_components, n_in_features)

    def test_invalid_manifold(self):
        """Test that invalid manifold raises error."""
        rngs = nnx.Rngs(42)

        with pytest.raises(ValueError, match="Unsupported manifold"):
            HoroPCA(
                n_components=5,
                n_in_features=10,
                manifold_name="invalid",
                c=1.0,
                rngs=rngs,
            )

    def test_to_hyperboloid_ideals_shape(self):
        """Test ideal point conversion shape."""
        n_components = 5
        n_in_features = 11
        c = 1.0
        rngs = nnx.Rngs(42)

        model = HoroPCA(
            n_components=n_components,
            n_in_features=n_in_features,
            manifold_name="hyperboloid",
            c=c,
            rngs=rngs,
        )

        # Create dummy orthonormal ideals
        ideals = jax.random.normal(jax.random.PRNGKey(42), (n_components, n_in_features - 1))
        ideals, _ = jnp.linalg.qr(ideals.T, mode="reduced")
        ideals = ideals.T

        # Convert to hyperboloid ideals
        hyperboloid_ideals = model._to_hyperboloid_ideals(ideals)

        assert hyperboloid_ideals.shape == (n_components, n_in_features)
        assert jnp.allclose(hyperboloid_ideals[:, 0], 1.0), "Temporal component should be 1"

    def test_fit_hyperboloid_basic(self):
        """Test basic fit functionality with hyperboloid."""
        key = jax.random.PRNGKey(42)
        n_points = 100
        n_in_features = 11
        n_components = 5
        c = 1.0

        # Generate random hyperboloid points
        x = jax.random.normal(key, (n_points, n_in_features))
        x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, c)

        # Create and fit model
        rngs = nnx.Rngs(42)
        model = HoroPCA(
            n_components=n_components,
            n_in_features=n_in_features,
            manifold_name="hyperboloid",
            c=c,
            lr=1e-3,
            max_steps=10,  # Small for quick test
            rngs=rngs,
        )

        # Fit should not raise errors
        model.fit(x)

        # Check that data_mean was computed
        assert model.data_mean[...] is not None
        assert model.data_mean[...].shape == (1, n_in_features)

    def test_fit_poincare_basic(self):
        """Test basic fit functionality with Poincaré ball."""
        key = jax.random.PRNGKey(42)
        n_points = 100
        n_in_features = 10
        n_components = 5
        c = 1.0

        # Generate random Poincaré points
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (n_points, n_in_features)) * 0.3

        # Create and fit model
        rngs = nnx.Rngs(42)
        model = HoroPCA(
            n_components=n_components,
            n_in_features=n_in_features,
            manifold_name="poincare",
            c=c,
            lr=1e-3,
            max_steps=10,  # Small for quick test
            rngs=rngs,
        )

        # Fit should not raise errors
        model.fit(x)

        # Check that data_mean was computed
        assert model.data_mean[...] is not None
        assert model.data_mean[...].shape == (1, n_in_features + 1)  # Hyperboloid representation

    def test_rank1_fit_transform_hyperboloid(self):
        """Ensure rank-1 configuration trains and transforms without NaNs."""
        key = jax.random.PRNGKey(0)
        n_points = 64
        n_in_features = 11
        c = 1.0

        x = jax.random.normal(key, (n_points, n_in_features))
        x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, c)

        rngs = nnx.Rngs(0)
        model = HoroPCA(
            n_components=1,
            n_in_features=n_in_features,
            manifold_name="hyperboloid",
            c=c,
            lr=1e-3,
            max_steps=8,
            rngs=rngs,
        )

        model.fit(x)
        assert model.loss_history[...].shape == (model.max_steps,)

        x_transformed = model.transform(x)
        assert x_transformed.shape == (n_points, 1)
        assert jnp.all(jnp.isfinite(x_transformed))

    def test_rank1_fit_transform_poincare(self):
        """Ensure rank-1 configuration works for Poincaré inputs."""
        key = jax.random.PRNGKey(1)
        n_points = 64
        n_in_features = 10
        c = 1.0

        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (n_points, n_in_features)) * 0.3

        rngs = nnx.Rngs(1)
        model = HoroPCA(
            n_components=1,
            n_in_features=n_in_features,
            manifold_name="poincare",
            c=c,
            lr=1e-3,
            max_steps=8,
            rngs=rngs,
        )

        model.fit(x)
        assert model.loss_history[...].shape == (model.max_steps,)

        x_transformed = model.transform(x)
        assert x_transformed.shape == (n_points, 1)
        assert jnp.all(jnp.isfinite(x_transformed))

    def test_transform_shape(self):
        """Test that transform produces correct output shape."""
        key = jax.random.PRNGKey(42)
        n_points = 100
        n_in_features = 11
        n_components = 5
        c = 1.0

        # Generate random hyperboloid points
        x = jax.random.normal(key, (n_points, n_in_features))
        x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, c)

        # Create, fit, and transform
        rngs = nnx.Rngs(42)
        model = HoroPCA(
            n_components=n_components,
            n_in_features=n_in_features,
            manifold_name="hyperboloid",
            c=c,
            lr=1e-3,
            max_steps=10,
            rngs=rngs,
        )

        model.fit(x)
        x_transformed = model.transform(x)

        assert x_transformed.shape == (n_points, n_components), (
            f"Expected shape ({n_points}, {n_components}), got {x_transformed.shape}"
        )

    def test_transform_without_fit_raises(self):
        """Test that transform without fit handles missing mean."""
        key = jax.random.PRNGKey(42)
        n_points = 100
        n_in_features = 11
        n_components = 5
        c = 1.0

        # Generate random hyperboloid points
        x = jax.random.normal(key, (n_points, n_in_features))
        x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, c)

        # Create model but don't fit
        rngs = nnx.Rngs(42)
        model = HoroPCA(
            n_components=n_components,
            n_in_features=n_in_features,
            manifold_name="hyperboloid",
            c=c,
            rngs=rngs,
        )

        # Transform should compute mean if recompute_mean=True
        x_transformed = model.transform(x, recompute_mean=True)
        assert x_transformed.shape == (n_points, n_components)

    def test_transform_preserves_approximate_distances(self):
        """Test that projection approximately preserves relative distances."""
        key = jax.random.PRNGKey(42)
        n_points = 50
        n_in_features = 11
        n_components = 8
        c = 1.0

        # Generate random hyperboloid points
        x = jax.random.normal(key, (n_points, n_in_features))
        x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, c)

        # Create and fit model
        rngs = nnx.Rngs(42)
        model = HoroPCA(
            n_components=n_components,
            n_in_features=n_in_features,
            manifold_name="hyperboloid",
            c=c,
            lr=1e-3,
            max_steps=50,
            rngs=rngs,
        )

        model.fit(x)
        x_transformed = model.transform(x)

        # Compute distances in original space (sample a few pairs)
        dist_orig_01 = hyperboloid.dist(x[0], x[1], c, hyperboloid.VERSION_DEFAULT)
        dist_orig_02 = hyperboloid.dist(x[0], x[2], c, hyperboloid.VERSION_DEFAULT)

        # Compute distances in transformed space (Poincaré ball)
        def poincare_dist(p1, p2):
            from hyperbolix.manifolds import poincare

            return poincare.dist(p1, p2, c, poincare.VERSION_MOBIUS_DIRECT)

        dist_trans_01 = poincare_dist(x_transformed[0], x_transformed[1])
        dist_trans_02 = poincare_dist(x_transformed[0], x_transformed[2])

        # Check that relative ordering is somewhat preserved
        # (This is a weak test since HoroPCA doesn't guarantee exact distance preservation)
        ratio_orig = dist_orig_01 / dist_orig_02
        ratio_trans = dist_trans_01 / dist_trans_02

        # Allow significant deviation since horospherical projection doesn't preserve distances exactly
        assert abs(jnp.log(ratio_orig / ratio_trans)) < 2.0, (
            f"Distance ratios differ too much: {ratio_orig:.4f} vs {ratio_trans:.4f}"
        )

    def test_compute_loss_decreases_during_optimization(self):
        """Test that loss generally decreases during optimization."""
        key = jax.random.PRNGKey(42)
        n_points = 100
        n_in_features = 11
        n_components = 5
        c = 1.0

        # Generate random hyperboloid points
        x = jax.random.normal(key, (n_points, n_in_features))
        x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, c)

        rngs = nnx.Rngs(42)
        model = HoroPCA(
            n_components=n_components,
            n_in_features=n_in_features,
            manifold_name="hyperboloid",
            c=c,
            lr=1e-2,
            max_steps=20,
            rngs=rngs,
        )

        model.fit(x)
        history = model.loss_history[...]
        assert history.shape == (model.max_steps,)
        assert jnp.all(jnp.isfinite(history))

        # Check that loss decreases on average (compare first and last quarters)
        # This is more robust than requiring monotonic decrease, since Adam
        # can temporarily increase loss due to momentum
        quarter_size = max(1, len(history) // 4)
        first_quarter_mean = jnp.mean(history[:quarter_size])
        last_quarter_mean = jnp.mean(history[-quarter_size:])

        # Allow small tolerance for numerical noise
        assert last_quarter_mean <= first_quarter_mean + 1e-3, (
            f"Loss should decrease on average: "
            f"first quarter mean={first_quarter_mean:.4f}, "
            f"last quarter mean={last_quarter_mean:.4f}"
        )


@pytest.mark.parametrize("manifold_name", ["hyperboloid", "poincare"])
@pytest.mark.parametrize("n_components", [3, 5, 7])
def test_horo_pca_different_dimensions(manifold_name, n_components):
    """Test HoroPCA with different dimensionalities."""
    key = jax.random.PRNGKey(42)
    n_points = 80
    c = 1.0

    if manifold_name == "hyperboloid":
        n_in_features = 11
        x = jax.random.normal(key, (n_points, n_in_features))
        x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, c)
    else:  # poincare
        n_in_features = 10
        x = jax.random.normal(key, (n_points, n_in_features)) * 0.3

    # Create and fit model
    rngs = nnx.Rngs(42)
    model = HoroPCA(
        n_components=n_components,
        n_in_features=n_in_features,
        manifold_name=manifold_name,
        c=c,
        lr=1e-3,
        max_steps=10,
        rngs=rngs,
    )

    model.fit(x)
    x_transformed = model.transform(x)

    assert x_transformed.shape == (n_points, n_components)
