import math
import torch

from .helpers import compute_mlr_PoincarePP, get_torch_dtype
from ..manifolds import ManifoldParameter, PoincareBall
from ..utils.math_utils import smooth_clamp, asinh


class HyperbolicRegressionPoincare(torch.nn.Module):
    """
    Module to compute the 'Hyperbolic Neural Networks' multinomial linear regression score(s):
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
    clamping_factor : float
        Clamping factor for the multinomial linear regression output (default: 1.0)
    smoothing_factor : float
        Smoothing factor for the multinomial linear regression output (default: 50.0)

    References
    ----------
    Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
        Advances in neural information processing systems 31 (2018).
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
        clamping_factor: float = 1.0,
        smoothing_factor: float = 50.0
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

        self.clamping_factor = clamping_factor
        self.smoothing_factor = smoothing_factor

    def _compute_mlr(self, x: torch.Tensor, a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Internal method for computing the multinomial linear regression score(s).

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
            The multinomial linear regression score(s) of x with respect to the linear model(s) defined by a and p.

        References
        ----------
        Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
            Advances in neural information processing systems 31 (2018).
        """
        x, a, p = self.manifold._2manifold_dtype([x, a, p])
        sqrt_c = self.manifold.c.sqrt()
        sub = self.manifold.addition(-p.T.unsqueeze(0), x.unsqueeze(-1), axis=1, backproject=self.backproject) # (B, in_dim, out_dim)
        suba = (sub * a.T).sum(dim=1, keepdim=True) # (B, 1, out_dim)
        a_norm = a.norm(p=2, dim=self.hyperbolic_axis, keepdim=True).clamp_min(self.manifold.min_enorm).T # (1, out_dim)
        asinh_arg = sqrt_c * self.manifold._lambda(sub, axis=1) * suba / a_norm # (B, 1, out_dim)
        # Improve the performance by smoothly clamping the input of asinh() to approximately the range of ...
        # ... [-16*clamping_factor, 16*clamping_factor] for float32
        # ... [-36*clamping_factor, 36*clamping_factor] for float64
        eps = torch.finfo(torch.float32).eps if self.manifold.dtype == torch.float32 else torch.finfo(torch.float64).eps
        clamp = self.clamping_factor * float(math.log(2 / eps))
        asinh_arg = smooth_clamp(asinh_arg, -clamp, clamp, self.smoothing_factor) # (B, out_dim)
        signed_dist2hyp = asinh(asinh_arg) / sqrt_c # (B, 1, out_dim)
        res = self.manifold._lambda(p, axis=self.hyperbolic_axis).T * a_norm * signed_dist2hyp.squeeze(1) # (B, out_dim)
        return res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x of shape (B, in_dim) where the hyperbolic_axis is last
        Parameters: self.weight of shape (out_dim, in_dim), self.bias of shape (out_dim, in_dim)
        Output: res of shape (B, out_dim)
        """
        assert self.manifold.is_in_manifold(self.bias, axis=self.hyperbolic_axis)

        if self.input_space == "tangent":
            x = self.manifold.expmap_0(x, axis=self.hyperbolic_axis, backproject=self.backproject)

        # Map self.weights from the tangent space at the origin to the tangent space at self.bias
        pt_weight = self.manifold.ptransp_0(self.weight, self.bias, axis=self.hyperbolic_axis, backproject=self.backproject)
        # Compute the multinomial linear regression score(s)
        res = self._compute_mlr(x, pt_weight, self.bias)
        return res

class HyperbolicRegressionPoincarePP(torch.nn.Module):
    """
    Module to compute the 'Hyperbolic Neural Networks ++' multinomial linear regression score(s):
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
    clamping_factor : float
        Clamping factor for the multinomial linear regression output (default: 1.0)
    smoothing_factor : float
        Smoothing factor for the multinomial linear regression output (default: 50.0)

    References
    ----------
    Shimizu Ryohei, Yusuke Mukuta, and Tatsuya Harada. "Hyperbolic neural networks++."
        arXiv preprint arXiv:2006.08210 (2020).
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
        clamping_factor: float = 1.0,
        smoothing_factor: float = 50.0
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
        bias = torch.zeros((self.output_dim, 1), dtype=self.params_dtype)
        self.bias = torch.nn.Parameter(bias, requires_grad=self.requires_grad)

        assert input_space in ["tangent", "manifold"], "input_space must be either 'tangent' or 'manifold'"
        self.input_space = input_space

        self.clamping_factor = clamping_factor
        self.smoothing_factor = smoothing_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x of shape (B, in_dim) where the hyperbolic_axis is last
        Parameters: self.weight of shape (out_dim, in_dim), self.bias of shape (out_dim, 1)
        Output: res of shape (B, out_dim)
        """
        if self.input_space == "tangent":
            x = self.manifold.expmap_0(x, axis=self.hyperbolic_axis, backproject=self.backproject)

        res = compute_mlr_PoincarePP(self.manifold, x, self.weight, self.bias,
                                     self.hyperbolic_axis, self.clamping_factor, self.smoothing_factor)
        return res
