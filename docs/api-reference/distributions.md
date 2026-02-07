# Distributions API

Probability distributions on hyperbolic manifolds.

## Overview

Hyperbolix provides wrapped normal distributions for probabilistic modeling on hyperbolic manifolds via functional interfaces. These distributions are essential for:

- Variational Autoencoders (VAEs) with hyperbolic latent spaces
- Bayesian neural networks on manifolds
- Uncertainty quantification in hyperbolic embeddings

## Wrapped Normal Distribution

The wrapped normal distribution extends the Gaussian distribution to hyperbolic manifolds by wrapping Euclidean Gaussians via the exponential map.

### Poincaré Wrapped Normal

::: hyperbolix.distributions.wrapped_normal_poincare
    options:
      show_source: true
      heading_level: 4

### Hyperboloid Wrapped Normal

::: hyperbolix.distributions.wrapped_normal_hyperboloid
    options:
      show_source: true
      heading_level: 4

## Usage Examples

### Basic Sampling (Poincaré)

```python
from hyperbolix.distributions import wrapped_normal_poincare
from hyperbolix.manifolds import poincare
import jax
import jax.numpy as jnp

# Mean on Poincaré ball
mean = jnp.array([0.2, 0.3])
mean_proj = poincare.proj(mean, c=1.0)

# Standard deviation
std = 0.1

# Sample
key = jax.random.PRNGKey(42)
samples = wrapped_normal_poincare.sample(mean_proj, std, c=1.0, key=key, sample_shape=(100,))
print(samples.shape)  # (100, 2)

# Samples lie on Poincaré ball
norms = jnp.linalg.norm(samples, axis=-1)
print(jnp.all(norms < 1.0 / jnp.sqrt(1.0)))  # True
```

### Log Probability

```python
# Compute log probability of samples
log_probs = jax.vmap(
    lambda x: wrapped_normal_poincare.log_prob(x, mean_proj, std, c=1.0)
)(samples)
print(log_probs.shape)  # (100,)

# Higher probability near mean
point_near_mean = poincare.proj(jnp.array([0.21, 0.29]), c=1.0)
point_far = poincare.proj(jnp.array([0.7, 0.7]), c=1.0)

print(f"Log prob (near): {wrapped_normal_poincare.log_prob(point_near_mean, mean_proj, std, c=1.0):.4f}")
print(f"Log prob (far): {wrapped_normal_poincare.log_prob(point_far, mean_proj, std, c=1.0):.4f}")
```

### Hyperboloid Distribution

```python
from hyperbolix.distributions import wrapped_normal_hyperboloid
from hyperbolix.manifolds import hyperboloid

# Mean on hyperboloid (ambient coordinates)
mean_space = jnp.array([0.2, 0.3, -0.1])
mean_ambient = jnp.concatenate([
    jnp.array([jnp.sqrt(jnp.sum(mean_space**2) + 1.0)]),
    mean_space
])

# Sample
key = jax.random.PRNGKey(123)
samples = wrapped_normal_hyperboloid.sample(mean_ambient, std=0.15, c=1.0, key=key, sample_shape=(50,))

# Compute log probabilities
log_probs = jax.vmap(
    lambda x: wrapped_normal_hyperboloid.log_prob(x, mean_ambient, 0.15, c=1.0)
)(samples)
```

## VAE Example

Using wrapped normal distributions in a Variational Autoencoder:

```python
from flax import nnx
from hyperbolix.distributions import wrapped_normal_poincare
from hyperbolix.nn_layers import HypLinearPoincare
from hyperbolix.manifolds import poincare
import jax
import jax.numpy as jnp

class HyperbolicVAE(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim

        # Encoder: Euclidean → Hyperbolic
        self.encoder = nnx.Linear(784, 128, rngs=rngs)
        self.enc_hyp = HypLinearPoincare(
            manifold_module=poincare,
            in_dim=128,
            out_dim=latent_dim,
            rngs=rngs
        )

        # Decoder: Hyperbolic → Euclidean
        self.dec_hyp = HypLinearPoincare(
            manifold_module=poincare,
            in_dim=latent_dim,
            out_dim=128,
            rngs=rngs
        )
        self.decoder = nnx.Linear(128, 784, rngs=rngs)

    def encode(self, x, c):
        # Returns mean and log_std for latent distribution
        h = jax.nn.relu(self.encoder(x))

        # Project to Poincaré ball
        h_proj = jax.vmap(poincare.proj, in_axes=(0, None, None))(h, c, None)

        # Mean on Poincaré ball
        mean = self.enc_hyp(h_proj, c)

        # Std in tangent space (Euclidean)
        log_std_layer = nnx.Linear(128, self.latent_dim, rngs=nnx.Rngs(0))
        log_std = log_std_layer(h)
        std = jnp.exp(log_std)

        return mean, std

    def decode(self, z, c):
        h = self.dec_hyp(z, c)

        # Logmap to tangent space for Euclidean decoder
        h_tangent = jax.vmap(poincare.logmap, in_axes=(None, 0, None))(
            jnp.zeros(self.latent_dim), h, c
        )

        return jax.nn.sigmoid(self.decoder(h_tangent))

    def __call__(self, x, key, c=1.0):
        # Encode
        mean, std = self.encode(x, c)

        # Sample latent code
        keys = jax.random.split(key, mean.shape[0])
        z = jax.vmap(
            lambda m, s, k: wrapped_normal_poincare.sample(m, s, c, k, ())
        )(mean, std, keys)

        # Decode
        recon = self.decode(z, c)

        return recon, mean, std, z

# Loss function
def vae_loss(model, x, key, c):
    recon, mean, std, z = model(x, key, c)

    # Reconstruction loss
    recon_loss = jnp.mean((x - recon) ** 2)

    # KL divergence (approximate for wrapped normal)
    # Use standard Gaussian prior in tangent space at origin
    kl_loss = -0.5 * jnp.mean(
        1 + 2 * jnp.log(std) - jnp.sum(mean**2, axis=-1) - std**2
    )

    return recon_loss + kl_loss
```

## Mathematical Background

### Wrapped Normal Definition

Given a mean $\mu \in \mathcal{M}$ on manifold $\mathcal{M}$ and standard deviation $\sigma$, the wrapped normal distribution is defined as:

1. Sample $v \sim \mathcal{N}(0, \sigma^2 I)$ in tangent space $T_\mu \mathcal{M}$
2. Wrap to manifold: $x = \exp_\mu(v)$

The log probability is:

$$
\log p(x) = -\frac{1}{2\sigma^2} \|\log_\mu(x)\|^2 - \frac{d}{2}\log(2\pi\sigma^2)
$$

where $\log_\mu$ is the logarithmic map at $\mu$.

### Sampling Algorithm

```python
def sample(mean, std, c, key, sample_shape):
    # 1. Sample in tangent space at mean
    tangent_sample = std * jax.random.normal(key, sample_shape + mean.shape)

    # 2. Exponential map to manifold
    manifold_sample = manifold.expmap(mean, tangent_sample, c)

    return manifold_sample
```

## Numerical Considerations

!!! warning "Numerical Stability"
    For small standard deviations and/or high curvatures, the exponential map can become numerically unstable. Consider:

    - Using float64 for very small $\sigma$ (< 0.01)
    - Clipping standard deviations to reasonable range: $\sigma \in [0.01, 1.0]$
    - Using version parameter in manifold operations for better stability

!!! tip "Curvature Choice"
    The curvature parameter $c$ affects the distribution:

    - Higher $c$ → More concentrated distributions
    - Lower $c$ → More spread out distributions

    Tune $c$ based on your application's needs.

## References

Wrapped distributions on manifolds are discussed in:

- Nagano, Y., et al. (2019). "A Wrapped Normal Distribution on Hyperbolic Space for Gradient-Based Learning"
- Davidson, T., et al. (2018). "Hyperspherical Variational Auto-Encoders"

See also:

- [Manifolds API](manifolds.md): Exponential and logarithmic maps
- [NN Layers API](nn-layers.md): Building VAEs with hyperbolic layers
