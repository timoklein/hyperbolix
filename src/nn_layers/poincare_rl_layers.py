import torch

from .helpers import get_torch_dtype
from ..manifolds import ManifoldParameter, PoincareBall
from ..utils.math_utils import asinh


class HyperbolicRegressionPoincareHDRL(torch.nn.Module):
    """
    Module to compute the 'Hyperbolic Deep Reinforcement Learning' multinomial linear regression score(s):
        0) Project the input tensor onto the manifold (optional)
        1) Compute the multinomial linear regression score(s)

    Parameters
    ----------
    manifold : PoincareBall
        The PoincareBall manifold
    input_dim : int
        Dimension of the input space
    output_dim : int
        Dimension of the output space
    hyperbolic_axis : int
        Axis along which the input tensor is hyperbolic (needs to be -1)
    backproject : bool
        Whether to project results back to the manifold (default: True)
    params_dtype : str
        Data type for the parameters (default: "float32")
    requires_grad : bool
        Whether the parameters should require gradients (default: True)
    input_space : str
        Type of the input tensor, either 'tangent' or 'manifold' (default: 'manifold')
    version : str
        Algorithm version to use: 'standard' or 'rs' (default: 'standard')
        - 'standard': scaled multinomial linear regression score function
        - 'rs': multinomial linear regression with parallel transported a

    References
    ----------
    Edoardo Cetin, Benjamin Chamberlain, Michael Bronstein and Jonathan J Hunt. "Hyperbolic deep reinforcement learning."
        arXiv (2022).
    Max Kochurov, Rasul Karimov and Serge Kozlukov. "Geoopt: Riemannian Optimization in PyTorch."
        arXiv (2020).
    """
    def __init__(
        self,
        manifold: PoincareBall,
        input_dim: int,
        output_dim: int,
        hyperbolic_axis: int = -1,
        backproject: bool = True,
        params_dtype: str = "float32",
        requires_grad: bool = True,
        input_space: str = "manifold",
        version: str = "standard"
    ):
        super().__init__()
        assert isinstance(manifold, PoincareBall), "manifold must be an instance of PoincareBall"
        assert hyperbolic_axis == -1, "hyperbolic_axis must be -1, reshape your tensor accordingly."
        self.manifold = manifold
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hyperbolic_axis = hyperbolic_axis
        self.backproject = backproject

        self.params_dtype = get_torch_dtype(params_dtype)
        if torch.finfo(self.params_dtype).eps < torch.finfo(manifold.dtype).eps:
            print(f"Warning: HyperbolicLayer.params_dtype is {self.params_dtype}, but Manifold.dtype is {manifold.dtype}."
                  f"All manifold operations will be performed in lower precision {manifold.dtype}!")

        self.requires_grad = requires_grad
        weight = torch.randn((output_dim, input_dim), dtype=self.params_dtype)
        self.weight = torch.nn.Parameter(weight, requires_grad=requires_grad)
        bias = torch.zeros((self.output_dim, self.input_dim), dtype=self.params_dtype)
        self.bias = ManifoldParameter(bias, requires_grad=self.requires_grad, manifold=self.manifold)

        assert input_space in ["tangent", "manifold"], "input_space must be either 'tangent' or 'manifold'"
        self.input_space = input_space

        assert version in ["standard", "rs"], "version must be either 'standard' or 'rs'"
        self.version = "forward_rs" if version == "rs" else "forward"

    def _dist2hyperplane(self, x: torch.Tensor, a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Computes the geodesic distance(s) of point(s) x to the hyperplane(s) defined by a and p.
        [Geoopt implementation]

        Parameters
        ----------
        x : torch.Tensor (B, in_dim)
            PoincareBall point(s)
        a : torch.Tensor (out_dim, in_dim)
            Hyperplane tangent normal(s) in the tangent space at p
        p : torch.Tensor (out_dim, in_dim)
            Hyperplane PoincareBall translation(s)

        Returns
        -------
        res : torch.Tensor (B, out_dim)
            The scaled signed geodesic distance(s) of x to the hyperplane(s) defined by a and p.

        References
        ----------
        Max Kochurov, Rasul Karimov and Serge Kozlukov. "Geoopt: Riemannian Optimization in PyTorch."
            arXiv (2020).
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).
        """
        ######################################################
        ### GEOOPT (k = -self.c, signed=True, scaled=True) ###
        # diff = _mobius_add(-p, x, k, dim=-1)
        # diff_norm2 = diff.pow(2).sum(dim=-1, keepdim=keepdim).clamp_min(1e-15)
        # sc_diff_a = (diff * a).sum(dim=-1, keepdim=keepdim)
        # a_norm = a.norm(dim=-1, keepdim=keepdim, p=2)
        # num = 2.0 * sc_diff_a
        # denom =  torch.abs((1 + k * diff_norm2) * a_norm) + 1e-15
        # distance = arsin_k(num / denom, k)  # geoopt uses asinh
        # distance = distance * a_norm
        # return distance
        ######################################################
        x, a, p = self.manifold._2manifold_dtype([x, a, p])
        sqrt_c = self.manifold.c.sqrt()
        diff = self.manifold.addition(-p, x, axis=-1, backproject=self.backproject) # (B, 1, out_dim, in_dim)
        diff_norm2 = diff.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-15) # (B, 1, out_dim, 1)
        sc_diff_a = (diff * a).sum(dim=-1, keepdim=True) # (B, 1, out_dim, 1)
        a_norm = a.norm(dim=-1, keepdim=True, p=2) # (out_dim, 1)
        num = 2.0 * sc_diff_a # (B, 1, out_dim, 1)
        denom = torch.abs((1 - self.manifold.c * diff_norm2) * a_norm) + 1e-15 # (B, 1, out_dim, 1)
        signed_distance = asinh(sqrt_c * num / denom) / sqrt_c # (B, 1, out_dim, 1)
        res = signed_distance * a_norm # (B, 1, out_dim, 1)
        return res

    def _compute_mlr(self, x: torch.Tensor, a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        HDRL multinomial linear regression implementation with dist2hyperplane from geoopt.

        Parameters
        ----------
        x : torch.Tensor (B, in_dim)
            PoincareBall point(s)
        a : torch.Tensor (out_dim, in_dim)
            Hyperplane tangent normal(s)
        p : torch.Tensor (out_dim, in_dim)
            Hyperplane PoincareBall translation(s)

        Returns
        -------
        res : torch.Tensor (B, out_dim)
            The (scaled) multinomial linear regression score(s) of x with respect to the linear model(s) defined by a and p.

        References
        ----------
        Edoardo Cetin, Benjamin Chamberlain, Michael Bronstein and Jonathan J Hunt. "Hyperbolic deep reinforcement learning."
            arXiv (2022)
        Max Kochurov, Rasul Karimov and Serge Kozlukov. "Geoopt: Riemannian Optimization in PyTorch."
            arXiv (2020).
        """
        x, a, p = self.manifold._2manifold_dtype([x, a, p])
        out_dim, in_dim = a.shape # out_dim, in_dim
        input_batch_dims = x.size()[:-1] # B if x is of shape (B, in_dim)
        input = x.view(-1, 1, in_dim) # (B, num_spaces=1, dimensions_per_space=in_dim)
        input_p = input.unsqueeze(-3) # (B, 1, num_spaces=1, dim_per_space=in_dim)

        if self.version == "forward":
            # Compute the scaled signed distance to the hyperplane. Scale=Euclidean norm instead of the tangent norm of a
            signed_distance = self._dist2hyperplane(input_p, a, p) # (B, 1, out_dim, 1)
            signed_distance = signed_distance #* self.logits_multiplier # logits_multiplier==1
        elif self.version == "forward_rs":
            # Parallel transport a to the tangent space at p and return the signed distance to the hyperplane (no scaling)
            conformal_factor = 1 - self.manifold.c * p.pow(2).sum(dim=-1, keepdim=True) # (out_dim, 1) # not actually the conformal factor
            signed_distance = self._dist2hyperplane(input_p, a*conformal_factor, p) # (B, 1, out_dim, 1)
            signed_distance = signed_distance * 2 / conformal_factor.view(1, 1, out_dim, 1) # (B, 1, out_dim, 1)

        signed_distance = signed_distance.sum(-1) # (B, 1, out_dim)
        res = signed_distance.view(*input_batch_dims, out_dim) # (B, num_planes=out_dim)
        return res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x of shape (B, in_dim) where the hyperbolic_axis is last
        Parameters: self.weight of shape (out_dim, in_dim), self.bias of shape (out_dim, in_dim)
        Output: res of shape (B, out_dim)
        """
        if self.input_space == "tangent":
            x = self.manifold.expmap_0(x, axis=self.hyperbolic_axis, backproject=self.backproject)

        # HDRL expands the weights to support multiple spaces at once
        # We don't use this. Instead we feed x of shape (B, on_dim)
        res = self._compute_mlr(x, self.weight, self.bias)
        return res
