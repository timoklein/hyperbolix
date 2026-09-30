# Distributions API

Probability distributions on hyperbolic manifolds.

## Overview

Hyperbolix provides wrapped normal distributions (Poincaré ball, hyperboloid) and a Riemannian-uniform distribution on a Poincaré geodesic ball, as plain functions. Uses include:

- Variational Autoencoders (VAEs) with hyperbolic latent spaces
- Bayesian neural networks on manifolds
- Uncertainty quantification in hyperbolic embeddings

## Riemannian Uniform Distribution

### Poincaré Uniform

::: hyperbolix.distributions.uniform_poincare
    options:
      show_source: true
      heading_level: 4

Samples uniformly with respect to the Riemannian volume measure within a geodesic ball $B(\text{center}, R)$ on the Poincaré ball. Uses geodesic polar decomposition: direction from $S^{n-1}$ (Muller method), radial component from $p(r) \propto \sinh^{n-1}(\sqrt{c}\,r)$.

- For $n=2$: closed-form radial sampling via $u = \cosh(\sqrt{c}\,r) - 1$
- For $n \geq 3$: rejection sampling with `jax.lax.while_loop` (JIT-compatible); for radii too small or too large for the loop, the radius is drawn in closed form.

**Usage:**

```python
from hyperbolix.distributions import uniform_poincare
from hyperbolix.manifolds import Poincare
import jax
import jax.numpy as jnp

poincare = Poincare()
key = jax.random.PRNGKey(42)

# Sample 100 points uniformly from geodesic ball B(origin, R=1.0) in 2D
# (n is the manifold dimension; the sample count goes in sample_shape)
samples = uniform_poincare.sample(key, n=2, c=1.0, R=1.0, sample_shape=(100,))
print(samples.shape)  # (100, 2)

# All samples inside the geodesic ball
inside = jax.vmap(poincare.dist_0, in_axes=(0, None))(samples, 1.0) <= 1.0
print(inside.all())  # True

# Volume of geodesic ball (2D closed-form: 2π(cosh R - 1)/c)
vol = uniform_poincare.volume(c=1.0, n=2, R=1.0)
print(vol)  # ~3.412 = 2π(cosh 1 − 1)

# Log probability (constant inside ball, -inf outside); batches natively over leading axes
log_p = uniform_poincare.log_prob(samples, c=1.0, R=1.0)
print(jnp.allclose(log_p, log_p[0]))  # True — uniform
```

**Use cases**: uniform priors for hyperbolic VAEs, and test points spread evenly over a geodesic ball.

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
from hyperbolix.manifolds import Poincare
import jax
import jax.numpy as jnp

poincare = Poincare()

# Mean on Poincaré ball
mean = jnp.array([0.2, 0.3])
mean_proj = poincare.proj(mean, c=1.0)

# Standard deviation
std = 0.1

# Sample
key = jax.random.PRNGKey(42)
samples = wrapped_normal_poincare.sample(
    key, mean_proj, std, c=1.0, sample_shape=(100,), manifold_module=poincare
)
print(samples.shape)  # (100, 2)

# Samples lie on Poincaré ball
norms = jnp.linalg.norm(samples, axis=-1)
print(jnp.all(norms < 1.0 / jnp.sqrt(1.0)))  # True
```

### Log Probability

```python
# Compute log probability of samples
log_probs = jax.vmap(
    lambda x: wrapped_normal_poincare.log_prob(x, mean_proj, std, c=1.0, manifold_module=poincare)
)(samples)
print(log_probs.shape)  # (100,)

# Higher probability near mean
point_near_mean = poincare.proj(jnp.array([0.21, 0.29]), c=1.0)
point_far = poincare.proj(jnp.array([0.7, 0.7]), c=1.0)

print(f"Log prob (near): {wrapped_normal_poincare.log_prob(point_near_mean, mean_proj, std, c=1.0, manifold_module=poincare):.4f}")
print(f"Log prob (far): {wrapped_normal_poincare.log_prob(point_far, mean_proj, std, c=1.0, manifold_module=poincare):.4f}")
```

### Hyperboloid Distribution

```python
from hyperbolix.distributions import wrapped_normal_hyperboloid
from hyperbolix.manifolds import Hyperboloid
import jax.numpy as jnp

hyperboloid = Hyperboloid()

# Mean on hyperboloid (ambient coordinates)
mean_space = jnp.array([0.2, 0.3, -0.1])
mean_ambient = jnp.concatenate([
    jnp.array([jnp.sqrt(jnp.sum(mean_space**2) + 1.0)]),
    mean_space
])

# Sample
key = jax.random.PRNGKey(123)
samples = wrapped_normal_hyperboloid.sample(
    key, mean_ambient, sigma=0.15, c=1.0, sample_shape=(50,), manifold_module=hyperboloid
)

# Compute log probabilities
log_probs = jax.vmap(
    lambda x: wrapped_normal_hyperboloid.log_prob(x, mean_ambient, 0.15, c=1.0, manifold_module=hyperboloid)
)(samples)
```

## VAE Example

A sketch of a Poincaré VAE: the encoder gives a tangent vector that `expmap_0` puts on the ball as
the mean, a second head gives the log standard deviation, the latent code is drawn with
`wrapped_normal_poincare.sample`, and the KL term is a one-sample Monte Carlo estimate from
`log_prob`.

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from hyperbolix.distributions import wrapped_normal_poincare
from hyperbolix.manifolds import Poincare

poincare = Poincare()
c = 1.0


class HyperbolicVAE(nnx.Module):
    def __init__(self, in_dim, latent_dim, rngs):
        self.encoder = nnx.Linear(in_dim, 128, rngs=rngs)
        self.mean_head = nnx.Linear(128, latent_dim, rngs=rngs)  # tangent vector at the origin
        self.log_std_head = nnx.Linear(128, latent_dim, rngs=rngs)  # trained with the rest
        self.decoder = nnx.Linear(latent_dim, in_dim, rngs=rngs)

    def __call__(self, x, key):
        h = jax.nn.relu(self.encoder(x))
        mean = jax.vmap(poincare.expmap_0, in_axes=(0, None))(self.mean_head(h), c)  # on the ball
        std = jnp.exp(self.log_std_head(h))  # per-axis standard deviation
        # A vector sigma is a per-axis std, so vmap over the batch (a 2-D sigma is a covariance)
        keys = jax.random.split(key, x.shape[0])
        z = jax.vmap(lambda k, m, s: wrapped_normal_poincare.sample(k, m, s, c))(keys, mean, std)
        recon = self.decoder(jax.vmap(poincare.logmap_0, in_axes=(0, None))(z, c))
        # One-sample Monte Carlo KL(q(z|x) || p(z)), prior = wrapped normal at the origin, sigma = 1
        log_q = jax.vmap(lambda z_, m, s: wrapped_normal_poincare.log_prob(z_, m, s, c))(z, mean, std)
        log_p = wrapped_normal_poincare.log_prob(z, jnp.zeros(z.shape[-1]), 1.0, c)
        return recon, log_q - log_p


def loss_fn(model, x, key):
    recon, kl = model(x, key)
    return jnp.mean(jnp.sum((x - recon) ** 2, axis=-1) + kl)


model = HyperbolicVAE(in_dim=784, latent_dim=2, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)


@nnx.jit
def train_step(model, optimizer, x, key):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, key)
    optimizer.update(model, grads)
    return loss


x = jax.random.uniform(jax.random.PRNGKey(1), (32, 784))  # stand-in for a data batch
print(train_step(model, optimizer, x, jax.random.PRNGKey(2)))
```

## Mathematical Background

### Wrapped Normal Definition

Given a mean $\mu \in \mathcal{M}$ and scale σ (a scalar or per-axis standard deviation, or an
$(n, n)$ covariance matrix $\Sigma$), a sample is drawn as follows:

1. Sample $v \sim \mathcal{N}(0, \Sigma)$ in the tangent space at the origin
2. Carry it to $\mu$ and wrap it onto the manifold: $x = \exp_\mu(\mathrm{PT}_{0\to\mu}(v))$

The log probability is:

$$
\log p(x) = \log \mathcal{N}(v;\,0,\Sigma) - (n-1)\log\frac{\sinh(\sqrt{c}\,r)}{\sqrt{c}\,r},\qquad v = \mathrm{PT}_{\mu\to 0}(\log_\mu x),\ r = \|v\|
$$

where $n$ is the manifold dimension and $r$ the Riemannian norm of $v$.

## Numerical Considerations

In float32, `log_prob` stays within ~1e-4 relative of float64 down to σ ≈ 1e-4; below that the
float32 rounding of the stored sample is a noticeable fraction of σ, so use float64.

!!! note "Curvature"
    For a fixed σ, the geodesic distance of a sample from μ has the same distribution for every
    $c$. The curvature changes only the ball coordinates (the ball has radius $1/\sqrt{c}$) and
    the volume correction in `log_prob`.

## References

Wrapped distributions on manifolds are discussed in:

- Nagano, Y., et al. (2019). "A Wrapped Normal Distribution on Hyperbolic Space for Gradient-Based Learning"
- Mathieu, E., et al. (2019). "Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders." NeurIPS.

See also:

- [Manifolds API](manifolds.md): Exponential and logarithmic maps
- [NN Layers API](nn-layers/vector-quantization.md): Building VAEs with hyperbolic VQ layers
