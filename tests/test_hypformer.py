"""Tests for Hyperbolic Transformation Component (HTC) and Hyperbolic Regularization Component (HRC)."""

import jax
import jax.numpy as jnp
from flax import nnx

from hyperbolix.manifolds import hyperboloid
from hyperbolix.nn_layers import (
    HRCBatchNorm,
    HRCDropout,
    HRCLayerNorm,
    HTCLinear,
    hrc,
    hrc_gelu,
    hrc_leaky_relu,
    hrc_relu,
    hrc_swish,
    hrc_tanh,
    htc,
)
from hyperbolix.nn_layers.hyperboloid_activations import (
    hyp_leaky_relu,
    hyp_relu,
    hyp_swish,
    hyp_tanh,
)

# Manifold constraint tests (most critical)


def test_hrc_manifold_constraint_same_curvature():
    """Test that HRC output satisfies hyperboloid constraint when c_in = c_out."""
    c = 1.0
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, c)

    y = hrc_relu(x, c_in=c, c_out=c)

    assert hyperboloid.is_in_manifold(y, c, atol=1e-5)


def test_hrc_manifold_constraint_different_curvature():
    """Test that HRC output satisfies hyperboloid constraint when c_in != c_out."""
    c_in = 1.0
    c_out = 2.0
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, c_in)

    y = hrc_relu(x, c_in=c_in, c_out=c_out)

    assert hyperboloid.is_in_manifold(y, c_out, atol=1e-5)


def test_hrc_manifold_constraint_batch():
    """Test manifold constraint for batched inputs."""
    c_in = 1.0
    c_out = 2.0
    batch_size = 32
    dim = 5

    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, dim))
    x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, c_in)

    y = hrc_relu(x, c_in=c_in, c_out=c_out)

    # Check each point in batch
    for i in range(batch_size):
        assert hyperboloid.is_in_manifold(y[i], c_out, atol=1e-5)


def test_htc_manifold_constraint_same_curvature():
    """Test that HTC output satisfies hyperboloid constraint when c_in = c_out."""
    c = 1.0
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, c)

    # Define a simple linear transformation
    W = jax.random.normal(jax.random.PRNGKey(0), (3, 4))

    def linear(z):
        return z @ W.T

    y = htc(x, linear, c_in=c, c_out=c)

    assert hyperboloid.is_in_manifold(y, c, atol=1e-5)


def test_htc_manifold_constraint_different_curvature():
    """Test that HTC output satisfies hyperboloid constraint when c_in != c_out."""
    c_in = 1.0
    c_out = 2.0
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, c_in)

    W = jax.random.normal(jax.random.PRNGKey(0), (3, 4))

    def linear(z):
        return z @ W.T

    y = htc(x, linear, c_in=c_in, c_out=c_out)

    assert hyperboloid.is_in_manifold(y, c_out, atol=1e-5)


# Equivalence tests


def test_hrc_equals_hyp_relu_when_curvatures_equal():
    """Test that HRC with ReLU matches existing hyp_relu when c_in = c_out."""
    c = 1.0
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, c)

    y_hrc = hrc_relu(x, c_in=c, c_out=c)
    y_hyp = hyp_relu(x, c)

    assert jnp.allclose(y_hrc, y_hyp, atol=1e-6)


def test_hrc_equals_hyp_tanh_when_curvatures_equal():
    """Test that HRC with tanh matches existing hyp_tanh when c_in = c_out."""
    c = 1.0
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, c)

    y_hrc = hrc_tanh(x, c_in=c, c_out=c)
    y_hyp = hyp_tanh(x, c)

    assert jnp.allclose(y_hrc, y_hyp, atol=1e-6)


def test_hrc_equals_hyp_leaky_relu_when_curvatures_equal():
    """Test that HRC with LeakyReLU matches existing hyp_leaky_relu when c_in = c_out."""
    c = 1.0
    negative_slope = 0.01
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, c)

    y_hrc = hrc_leaky_relu(x, c_in=c, c_out=c, negative_slope=negative_slope)
    y_hyp = hyp_leaky_relu(x, c, negative_slope=negative_slope)

    assert jnp.allclose(y_hrc, y_hyp, atol=1e-6)


def test_hrc_equals_hyp_swish_when_curvatures_equal():
    """Test that HRC with Swish matches existing hyp_swish when c_in = c_out."""
    c = 1.0
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, c)

    y_hrc = hrc_swish(x, c_in=c, c_out=c)
    y_hyp = hyp_swish(x, c)

    assert jnp.allclose(y_hrc, y_hyp, atol=1e-6)


def test_hrc_equals_hyp_activations_batch():
    """Test equivalence with existing activations for batched inputs."""
    c = 1.0
    batch_size = 32
    dim = 5

    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, dim))
    x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, c)

    # Test ReLU
    y_hrc = hrc_relu(x, c_in=c, c_out=c)
    y_hyp = hyp_relu(x, c)
    assert jnp.allclose(y_hrc, y_hyp, atol=1e-6)

    # Test tanh
    y_hrc = hrc_tanh(x, c_in=c, c_out=c)
    y_hyp = hyp_tanh(x, c)
    assert jnp.allclose(y_hrc, y_hyp, atol=1e-6)


# Shape tests


def test_hrc_shape_single():
    """Test HRC preserves shape for single point."""
    c = 1.0
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, c)

    y = hrc_relu(x, c_in=c, c_out=c)

    assert y.shape == x.shape


def test_hrc_shape_batch():
    """Test HRC shape for batched inputs."""
    c = 1.0
    batch_size = 32
    dim = 5

    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, dim))
    x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, c)

    y = hrc_relu(x, c_in=c, c_out=2.0)

    assert y.shape == x.shape


def test_hrc_shape_multi_dim():
    """Test HRC with multi-dimensional batch (e.g., feature maps)."""
    c = 1.0
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 16, 10))
    x = jax.vmap(
        jax.vmap(jax.vmap(hyperboloid.proj, in_axes=(0, None)), in_axes=(0, None)),
        in_axes=(0, None),
    )(x, c)

    y = hrc_relu(x, c_in=c, c_out=c)

    assert y.shape == x.shape


def test_htc_shape_change():
    """Test HTC with dimension change (in_dim != out_dim)."""
    c = 1.0
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, c)

    # Transform from 4D to 3D spatial (output will be 4D: 3 spatial + 1 time)
    W = jax.random.normal(jax.random.PRNGKey(0), (3, 4))

    def linear(z):
        return z @ W.T

    y = htc(x, linear, c_in=c, c_out=c)

    assert y.shape == (4,)  # 3 spatial + 1 time


# Gradient tests


def test_hrc_gradients_finite():
    """Test that gradients through HRC are finite."""
    c_in = 1.0
    c_out = 2.0
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, c_in)

    def loss_fn(x_input):
        y = hrc_relu(x_input, c_in=c_in, c_out=c_out)
        return jnp.sum(y**2)

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(x)

    assert jnp.all(jnp.isfinite(grads))


def test_htc_gradients_finite():
    """Test that gradients through HTC are finite."""
    c_in = 1.0
    c_out = 2.0
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, c_in)

    W = jax.random.normal(jax.random.PRNGKey(0), (3, 4))

    def loss_fn(x_input):
        y = htc(x_input, lambda z: z @ W.T, c_in=c_in, c_out=c_out)
        return jnp.sum(y**2)

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(x)

    assert jnp.all(jnp.isfinite(grads))


def test_gradients_wrt_curvature():
    """Test gradients with respect to curvature parameters."""
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, 1.0)

    def loss_fn(c_out_val):
        y = hrc_relu(x, c_in=1.0, c_out=c_out_val)
        return jnp.sum(y**2)

    grad_fn = jax.grad(loss_fn)
    grad_c = grad_fn(2.0)

    assert jnp.isfinite(grad_c)


# JIT compatibility tests


def test_hrc_jit():
    """Test that HRC works under JIT compilation."""

    @jax.jit
    def jitted_hrc(x, c_in, c_out):
        return hrc_relu(x, c_in, c_out)

    c_in = 1.0
    c_out = 2.0
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, c_in)

    y = jitted_hrc(x, c_in, c_out)

    assert hyperboloid.is_in_manifold(y, c_out, atol=1e-5)


def test_htc_jit():
    """Test that HTC works under JIT compilation."""
    W = jax.random.normal(jax.random.PRNGKey(0), (3, 4))

    @jax.jit
    def jitted_htc(x, c_in, c_out):
        return htc(x, lambda z: z @ W.T, c_in, c_out)

    c_in = 1.0
    c_out = 2.0
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, c_in)

    y = jitted_htc(x, c_in, c_out)

    assert hyperboloid.is_in_manifold(y, c_out, atol=1e-5)


def test_hrc_module_nnx_jit():
    """Test that HRC NNX modules work under nnx.jit."""
    dropout = HRCDropout(rate=0.1, rngs=nnx.Rngs(dropout=42))

    @nnx.jit
    def forward(model, x, c_in, c_out):
        return model(x, c_in, c_out, deterministic=True)

    c_in = 1.0
    c_out = 2.0
    x = jnp.ones((8, 5))
    x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, c_in)

    y = forward(dropout, x, c_in, c_out)

    assert y.shape == x.shape


# Edge case tests


def test_hrc_zero_spatial_components():
    """Test HRC when spatial components become zero after activation."""
    c_in = 1.0
    c_out = 2.0

    # Create point with negative spatial components that ReLU will zero out
    x = jnp.array([1.0 / jnp.sqrt(c_in), -0.1, -0.2, -0.15])

    y = hrc_relu(x, c_in=c_in, c_out=c_out)

    # Should be close to origin on output manifold
    expected_origin = jnp.array([1.0 / jnp.sqrt(c_out), 0.0, 0.0, 0.0])
    assert jnp.allclose(y, expected_origin, atol=1e-6)


def test_htc_zero_output():
    """Test HTC when linear transformation produces near-zero output."""
    c_in = 1.0
    c_out = 2.0
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, c_in)

    # Zero transformation
    def zero_transform(z):
        return jnp.zeros(3)

    y = htc(x, zero_transform, c_in=c_in, c_out=c_out)

    # Should be close to origin on output manifold
    expected_origin = jnp.array([1.0 / jnp.sqrt(c_out), 0.0, 0.0, 0.0])
    assert jnp.allclose(y, expected_origin, atol=1e-6)


def test_curvature_ratio_extreme_values():
    """Test HRC with extreme curvature ratios."""
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, 1.0)

    # Very small to very large curvature
    y = hrc_relu(x, c_in=0.1, c_out=10.0)
    assert hyperboloid.is_in_manifold(y, 10.0, atol=1e-4)

    # Very large to very small curvature
    x = hyperboloid.proj(x, 10.0)
    y = hrc_relu(x, c_in=10.0, c_out=0.1)
    assert hyperboloid.is_in_manifold(y, 0.1, atol=1e-4)


# NNX Module tests


def test_htc_linear_forward():
    """Test HTCLinear forward pass."""
    layer = HTCLinear(in_features=5, out_features=8, rngs=nnx.Rngs(0))

    x = jnp.ones((32, 5))
    x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, 1.0)

    y = layer(x, c_in=1.0, c_out=2.0)

    assert y.shape == (32, 9)  # 8 spatial + 1 time
    for i in range(32):
        assert hyperboloid.is_in_manifold(y[i], 2.0, atol=1e-5)


def test_htc_linear_gradient():
    """Test gradients through HTCLinear."""
    layer = HTCLinear(in_features=5, out_features=8, rngs=nnx.Rngs(0))

    x = jnp.ones((32, 5))
    x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, 1.0)

    def loss_fn(model):
        y = model(x, c_in=1.0, c_out=2.0)
        return jnp.sum(y**2)

    loss, grads = nnx.value_and_grad(loss_fn)(layer)

    assert jnp.isfinite(loss)
    # Check that gradients exist and are finite
    assert hasattr(grads, "kernel")
    assert jnp.all(jnp.isfinite(grads.kernel[...]))


def test_hrc_dropout_training_vs_eval():
    """Test HRCDropout behaves differently in training vs eval mode."""
    dropout = HRCDropout(rate=0.5, rngs=nnx.Rngs(dropout=42))

    x = jnp.ones((32, 5))
    x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, 1.0)

    # Training mode (stochastic)
    y_train = dropout(x, c_in=1.0, c_out=1.0, deterministic=False)

    # Eval mode (deterministic)
    y_eval = dropout(x, c_in=1.0, c_out=1.0, deterministic=True)

    # In eval mode, output should match input (no dropout)
    assert jnp.allclose(y_eval, x, atol=1e-6)

    # In training mode, some values should be different (dropped out)
    # Note: This test might rarely fail due to randomness
    assert not jnp.allclose(y_train, x, atol=1e-6)


def test_hrc_layernorm_forward():
    """Test HRCLayerNorm forward pass."""
    ln = HRCLayerNorm(num_features=4, rngs=nnx.Rngs(0))

    x = jnp.ones((32, 5))
    x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, 1.0)

    y = ln(x, c_in=1.0, c_out=2.0)

    assert y.shape == x.shape
    for i in range(32):
        assert hyperboloid.is_in_manifold(y[i], 2.0, atol=1e-5)


# Additional activation tests


def test_hrc_gelu():
    """Test HRC with GELU activation."""
    c_in = 1.0
    c_out = 2.0
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, c_in)

    y = hrc_gelu(x, c_in=c_in, c_out=c_out)

    assert hyperboloid.is_in_manifold(y, c_out, atol=1e-5)


def test_hrc_with_custom_function():
    """Test HRC with a custom function."""
    c = 1.0
    x = jnp.array([1.05, 0.1, -0.2, 0.15])
    x = hyperboloid.proj(x, c)

    def custom_activation(z):
        return jax.nn.gelu(z) * 0.5

    y = hrc(x, custom_activation, c_in=c, c_out=c)

    assert hyperboloid.is_in_manifold(y, c, atol=1e-5)


# Batch consistency tests


def test_hrc_batch_consistency():
    """Test that batched HRC gives same results as individual applications."""
    c_in = 1.0
    c_out = 2.0

    # Create batch of points
    batch_size = 8
    x_batch = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 5))
    x_batch = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x_batch, c_in)

    # Apply HRC to batch
    y_batch = hrc_relu(x_batch, c_in=c_in, c_out=c_out)

    # Apply HRC to individual points
    for i in range(batch_size):
        y_single = hrc_relu(x_batch[i], c_in=c_in, c_out=c_out)
        assert jnp.allclose(y_batch[i], y_single, atol=1e-6)


# HRCBatchNorm tests


def test_hrc_batchnorm_forward():
    """Test HRCBatchNorm forward pass with manifold constraint."""
    bn = HRCBatchNorm(num_features=4, rngs=nnx.Rngs(0))

    x = jax.random.normal(jax.random.PRNGKey(42), (32, 5))
    x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, 1.0)

    # Training mode
    y = bn(x, c_in=1.0, c_out=1.0, use_running_average=False)

    assert y.shape == x.shape
    for i in range(32):
        assert hyperboloid.is_in_manifold(y[i], 1.0, atol=1e-5)


def test_hrc_batchnorm_curvature_change():
    """Test HRCBatchNorm with curvature change."""
    bn = HRCBatchNorm(num_features=4, rngs=nnx.Rngs(0))

    c_in = 1.0
    c_out = 2.0
    x = jax.random.normal(jax.random.PRNGKey(42), (32, 5))
    x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, c_in)

    y = bn(x, c_in=c_in, c_out=c_out, use_running_average=False)

    assert y.shape == x.shape
    for i in range(32):
        assert hyperboloid.is_in_manifold(y[i], c_out, atol=1e-5)


def test_hrc_batchnorm_training_vs_eval():
    """Test HRCBatchNorm behaves differently in training vs eval mode."""
    bn = HRCBatchNorm(num_features=4, rngs=nnx.Rngs(0))

    x = jax.random.normal(jax.random.PRNGKey(42), (32, 5))
    x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, 1.0)

    # Training mode - update running statistics
    y_train1 = bn(x, c_in=1.0, c_out=1.0, use_running_average=False)
    y_train2 = bn(x, c_in=1.0, c_out=1.0, use_running_average=False)

    # Evaluation mode - use running statistics
    y_eval = bn(x, c_in=1.0, c_out=1.0, use_running_average=True)

    # All outputs should be valid manifold points
    for i in range(32):
        assert hyperboloid.is_in_manifold(y_train1[i], 1.0, atol=1e-5)
        assert hyperboloid.is_in_manifold(y_train2[i], 1.0, atol=1e-5)
        assert hyperboloid.is_in_manifold(y_eval[i], 1.0, atol=1e-5)

    # Eval mode output should be different from training mode
    # (using running average vs batch statistics)
    assert not jnp.allclose(y_eval, y_train1, atol=1e-6)


def test_hrc_batchnorm_shape_preservation():
    """Test HRCBatchNorm preserves input shape."""
    bn = HRCBatchNorm(num_features=16, rngs=nnx.Rngs(0))

    # Test with different batch sizes
    for batch_size in [8, 16, 32]:
        x = jax.random.normal(jax.random.PRNGKey(batch_size), (batch_size, 17))
        x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, 1.0)

        y = bn(x, c_in=1.0, c_out=1.0, use_running_average=False)

        assert y.shape == x.shape


def test_hrc_batchnorm_gradient():
    """Test gradients through HRCBatchNorm."""
    bn = HRCBatchNorm(num_features=4, rngs=nnx.Rngs(0))

    x = jax.random.normal(jax.random.PRNGKey(42), (32, 5))
    x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, 1.0)

    def loss_fn(model):
        y = model(x, c_in=1.0, c_out=2.0, use_running_average=False)
        return jnp.sum(y**2)

    loss, grads = nnx.value_and_grad(loss_fn)(bn)

    assert jnp.isfinite(loss)
    # Check that gradients exist for scale and bias parameters
    assert hasattr(grads.bn, "scale")
    assert hasattr(grads.bn, "bias")
    assert jnp.all(jnp.isfinite(grads.bn.scale[...]))
    assert jnp.all(jnp.isfinite(grads.bn.bias[...]))


def test_hrc_batchnorm_jit():
    """Test HRCBatchNorm works under JIT compilation."""
    bn = HRCBatchNorm(num_features=4, rngs=nnx.Rngs(0))

    @nnx.jit
    def forward(model, x, c_in, c_out):
        return model(x, c_in, c_out, use_running_average=False)

    x = jax.random.normal(jax.random.PRNGKey(42), (32, 5))
    x = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, 1.0)

    y = forward(bn, x, 1.0, 2.0)

    assert y.shape == x.shape
    for i in range(32):
        assert hyperboloid.is_in_manifold(y[i], 2.0, atol=1e-5)


def test_hrc_batchnorm_extreme_curvatures():
    """Test HRCBatchNorm with extreme curvature ratios."""
    bn = HRCBatchNorm(num_features=4, rngs=nnx.Rngs(0))

    x = jax.random.normal(jax.random.PRNGKey(42), (32, 5))

    # Very small to very large curvature
    x_small = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, 0.1)
    y = bn(x_small, c_in=0.1, c_out=10.0, use_running_average=False)

    for i in range(32):
        assert hyperboloid.is_in_manifold(y[i], 10.0, atol=1e-4)

    # Very large to very small curvature
    x_large = jax.vmap(hyperboloid.proj, in_axes=(0, None))(x, 10.0)
    y = bn(x_large, c_in=10.0, c_out=0.1, use_running_average=False)

    for i in range(32):
        assert hyperboloid.is_in_manifold(y[i], 0.1, atol=1e-4)
