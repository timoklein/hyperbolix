import torch
import torch.nn as nn

from .helpers import compute_pairwise_distances
from ..manifolds import Manifold, PoincareBall, Hyperboloid


def compute_frechet_mean(x: torch.Tensor, hyperboloid: Hyperboloid) -> torch.Tensor:
    """
    Compute the Frechet mean of the Hyperboloid point(s) x and save it as the data_mean attribute.

    Parameters
    ----------
    x : torch.Tensor
        Hyperboloid point(s)
    hyperboloid : Hyperboloid
        The Hyperboloid manifold

    Returns
    -------
    mean : torch.Tensor (dtype=hyperboloid.dtype)
        The Frechet mean
    """
    # Set the inital mean to be the centroid of the squared Lorentzian distance
    # Note: This must not necessarily be a minimizer of the geodesic distance
    x_sum = x.sum(dim=0, keepdim=True)
    denom = (hyperboloid.c * torch.abs(hyperboloid._minkowski_inner(x_sum, x_sum))).sqrt()
    mean_init = x_sum / denom
    mean_init = hyperboloid.proj(mean_init)
    has_converged = False
    batch_size = x.shape[0]
    # Try multiple learning rates
    for lr in [1e-02, 2e-02, 5e-03, 4e-02, 2.5e-03]:
        mean = mean_init
        for _ in range(5_000):
            # Compute the logarithmic map of x with respect to the current mean
            log_x = torch.sum(hyperboloid.logmap(x, mean), dim=0, keepdim=True)
            update = lr * log_x / batch_size
            # Update the mean using the exponential map
            mean = hyperboloid.expmap(update, mean)
            # Stop if the update has become negligible
            if update.norm(p=2, dim=-1, keepdim=False) < 5e-06:
                has_converged = True
                break
        if has_converged:
            break
    else:
        # If neither learning rate suceeded take the best candidate mean
        print("compute_frechet_mean: No convergence with any learning rate. "
              "Using the best candidate mean.", flush=True)
    return mean

def center_data(x: torch.Tensor, mean: torch.Tensor, hyperboloid: Hyperboloid) -> torch.Tensor:
    """
    Center Hyperboloid point(s) around their Frechet mean.

    Parameters
    ----------
    x : torch.Tensor
        Hyperboloid point(s)
    mean : torch.Tensor
        The Frechet mean of x
    hyperboloid : Hyperboloid
        The Hyperboloid manifold

    Returns
    -------
    res : torch.Tensor (dtype=hyperboloid.dtype)
        The centered Hyperboloid point(s)
    """
    # 1) Compute the Lorentz transformation that maps the mean to the Hyperboloid's origin
    gamma = mean[:,:1] * hyperboloid.c.sqrt()
    velocity = mean[:,1:] / mean[:,:1]
    # Compute the individual blocks of the (transposed) transformation matrix
    block_tl = gamma # top-left block: gamma (scalar)
    block_tr = -gamma * velocity # top-right block: -gamma * v
    top_row = torch.cat((block_tl, block_tr), dim=1)
    block_bl = block_tr.T # bottom-left block: -gamma * v.T
    identity_n = torch.eye(velocity.shape[1], dtype=x.dtype, device=x.device)
    vTv = velocity.T @ velocity
    coefficient = (gamma**2) / (1 + gamma)
    block_br = identity_n + coefficient * vTv # bottom-right block: I + (gamma^2 / (1 + gamma)) * v.T v
    bottom_row = torch.cat((block_bl, block_br), dim=1)
    lorentz_boost = torch.cat((top_row, bottom_row), dim=0)
    # 2) Apply the Lorentz transformation to the data
    res = x @ lorentz_boost
    # 3) Backproject the data to the Hyperboloid
    res = hyperboloid.proj(res)
    return res

class HoroPCA(nn.Module):
    """
    Horospherical projections dimensionality reduction class.
    Reimplmentation of the HoroPCA method with adjustments for stability, speed, and curvature.

    References
    ----------
    Ines Chami, et al. "Horopca: Hyperbolic dimensionality reduction via horospherical projections."
        International Conference on Machine Learning (2021).
    Weize Chen, et al. "Fully hyperbolic neural networks."
        arXiv preprint arXiv:2105.14686 (2021).
    """
    def __init__(
        self,
        n_components: int,
        n_in_features: int,
        manifold: Manifold,
        lr: float = 1e-3,
        max_steps: int = 100,
    ):
        super().__init__()
        self.n_components = n_components
        self.n_in_features = n_in_features
        self.manifold = manifold
        self.lr = lr
        self.max_steps = max_steps
        self.data_mean = None
        # Initialize the manifolds for horo projection and the principal components (ideal points)
        if isinstance(self.manifold, PoincareBall):
            self.hyperboloid = Hyperboloid(c=self.manifold.c, dtype=self.manifold.dtype)
            self.Q = nn.Parameter(torch.randn(self.n_components, self.n_in_features,
                                              dtype=self.manifold.dtype, device=self.manifold.c.device))
        elif isinstance(self.manifold, Hyperboloid):
            self.hyperboloid = self.manifold
            self.Q = nn.Parameter(torch.randn(self.n_components, self.n_in_features-1,
                                              dtype=self.manifold.dtype, device=self.manifold.c.device))
        else:
            raise ValueError("Unsupported manifold type. Use PoincareBall or Hyperboloid.")

    def _to_hyperboloid_ideals(self, ideals: torch.Tensor) -> torch.Tensor:
        """
        Convert the orthonormalized PoincareBall ideal point(s) to Hyperboloid ideal point(s).
        Ideal points in the Hyperboloid are represented by the directions of the corresponding
        1-dimensional null cones. Hence the PoincareBall ideal points need not lie in the closure
        of the PoincareBall. They only need to be orthonormalized to conform with the mapping below.

        Parameters
        ----------
        x : torch.Tensor
            Orthonormalized PoincareBall ideal point(s)

        Returns
        -------
        res : torch.Tensor
            The Hyperboloid ideal point(s)
        """
        res = torch.cat([torch.ones_like(ideals[:,:1]), ideals], dim=-1)
        return res

    def _horo_projection(self, x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Compute the horospherical projection(s) based on horosphere intersections in the Hyperboloid.
        The target submanifold has dimension self.n_components and is a geodesic submanifold passing through
        the Hyperboloid's ideal points and its origin (1/sqrt(c),0,0,0,...). The geodesic submanifold spanned
        by the ideals must not contain the Hyperboloid's origin and the ideals have to be linearly independent.

        Parameters
        ----------
        x : torch.Tensor
            Hyperboloid point(s)
        Q : torch.Tensor
            Ideal point(s) in the Hyperboloid with orthonormalized space coordinates

        Returns
        -------
        res : torch.Tensor (dtype=self.manifold.dtype)
            The horospherical projection(s) of x
        """
        # Compute the orthogonal geodesic projection [x B Q^T (Q B Q^T)^-1 Q] of x onto the geodesic
        # submanifold ("spine") spanned by the ideals ("open book" interpretation), where B is the
        # Minkowski inner product matrix [batched version of Prop. A.23.2 in the HoroPCA paper]
        # 1) Compute the coefficients [x_coeffs = x B Q^T (Q B Q^T)^-1] of the projection.
        #    Since the space coordinates of Q are normalized we can solve the linear system directly
        #    using the Sherman–Morrison formula to compute (Q B Q^T)^-1. The matrix to be inverted here
        #    is just the identity matrix plus the outer product between [-1,...,-1] and [1,...,1].T.
        xBQt = self.hyperboloid._minkowski_inner(x.unsqueeze(-1), Q.T.unsqueeze(0), axis=1).squeeze(1)
        QBQt_inverse = (torch.eye(self.n_components, device=x.device, dtype=x.dtype) + 1/(1-self.n_components))
        x_coeffs = xBQt @ QBQt_inverse
        # 2) Compute the orthogonal geodesic projection onto the spine
        mink_proj = x_coeffs @ Q
        mink_proj_normalized = (-self.hyperboloid.c * self.hyperboloid._minkowski_inner(mink_proj, mink_proj)).sqrt()
        spine_proj = mink_proj / mink_proj_normalized
        # Compute the tangent vectors of the hyperboloid with base point spine_proj that are pointing
        # towards hyperboloid_origin, are tangent to the target submanifold, and are orthogonal to the spine
        # Note: We orthogonalize the origin to the spine instead of the chords to save compute
        hyperboloid_origin = self.hyperboloid._create_origin_from_reference(spine_proj)
        originBQt = self.hyperboloid._minkowski_inner(hyperboloid_origin.unsqueeze(-1), Q.T.unsqueeze(0), axis=1).squeeze(1)
        origin_coeffs = originBQt @ QBQt_inverse
        tangents = hyperboloid_origin - (origin_coeffs @ Q)
        # Assign the tangent vectors the correct speed such that by mapping them to the Hyperboloid via the exponential map
        # the horospherical projection of x is at distance 'spine_dist' apart from the original point x
        unit_tangents = tangents / self.hyperboloid._minkowski_inner(tangents, tangents).sqrt()
        tangents = self.hyperboloid.dist(x, spine_proj) * unit_tangents
        res = self.hyperboloid.expmap(tangents, spine_proj)
        return res

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the generalized variance of the projected point(s).

        Parameters
        ----------
        x : torch.Tensor
            Manifold point(s)

        Returns
        -------
        -var : torch.Tensor (dtype=self.manifold.dtype)
            The negative generalized variance of the projected point(s)
        """
        # Orthonormalize the principal components
        Q_ortho, _ = torch.linalg.qr(self.Q.T, mode='reduced')
        # Map the principal components to the null cone
        hyperboloid_ideals = self._to_hyperboloid_ideals(Q_ortho.T)
        # Project x onto the submanifold spanned by the Hyperboloid's principal components
        x_proj = self._horo_projection(x, hyperboloid_ideals)
        # Compute the (smoothened) pairwise distances directly in the Hyperboloid
        distances = compute_pairwise_distances(x_proj, self.hyperboloid, version="smoothened")
        # Compute the biased generalized variance of the projected points
        var = torch.mean(distances ** 2)
        return -var

    @torch.enable_grad()
    def fit(self, x: torch.Tensor) -> None:
        """
        Find the principal component(s) using gradient-descent-based optimization.

        Parameters
        ----------
        x : torch.Tensor
            Manifold point(s) of shape (n_samples, n_in_features)
        """
        if isinstance(self.manifold, PoincareBall):
            x = self.manifold.to_hyperboloid(x)
        # Compute the Frechet mean of the data points
        self.data_mean = compute_frechet_mean(x, self.hyperboloid)
        # Center the data points around their Frechet mean
        x_centered = center_data(x, self.data_mean, self.hyperboloid)
        # The parameters of the model are ideal points that lie in the manifold's closure, i.e. they
        # are part of the Euclidean ambient space and do not lie in the hyperbolic space itself
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        # Iteratively compute the projected variance loss and update the parameters
        for _ in range(self.max_steps):
            optim.zero_grad()
            loss = self.compute_loss(x_centered)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1e05)
            optim.step()

    def transform(self, x: torch.Tensor, recompute_mean: bool=False) -> torch.Tensor:
        """
        Project the point(s) x onto the submanifold containing the origin that is
        spanned by the generalized principal components of the PoincareBall.

        Parameters
        ----------
        x : torch.Tensor
            Manifold point(s) of shape (n_samples, self.n_in_features)
        recompute_mean : bool (optional)
            If True, recompute the Frechet mean of the point(s) x (default: False)

        Returns
        -------
        res : torch.Tensor (dtype=self.manifold.dtype)
            The projected PoincareBall point(s) of shape (n_samples, self.n_components)
        """
        if isinstance(self.manifold, PoincareBall):
            x = self.manifold.to_hyperboloid(x)
        if recompute_mean or self.data_mean is None:
            self.data_mean = compute_frechet_mean(x, self.hyperboloid)
        # Center the data points around their Frechet mean
        x_centered = center_data(x, self.data_mean, self.hyperboloid)
        # Orthonormalize the principal components
        Q_ortho, _ = torch.linalg.qr(self.Q.T, mode='reduced')
        # Map the principal components to the null cone
        hyperboloid_ideals = self._to_hyperboloid_ideals(Q_ortho.T)
        # Project x onto the submanifold spanned by the Hyperboloid's principal components
        x_proj = self._horo_projection(x_centered, hyperboloid_ideals)
        # Map the projected points back to the PoincareBall
        x_poincare = self.hyperboloid.to_poincare(x_proj)
        # Compute the coordinates in the lower-dimensional PoincareBall
        res = x_poincare @ Q_ortho
        return res
