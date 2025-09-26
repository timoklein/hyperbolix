import torch

from .helpers import compute_mlr_PoincarePP, get_torch_dtype
from ..manifolds import ManifoldParameter, PoincareBall
from ..utils.math_utils import sinh


class HyperbolicLinearPoincare(torch.nn.Module):
    """
    Module to compute the 'Hyperbolic Neural Networks' fully connected layer:
        0) Project the input tensor to the tangent space (optional)
        1) Perform matrix vector multiplication in the tangent space at the origin.
        2) Map the result to the manifold.
        3) Add the manifold bias to the result.

    Parameters
    ----------
    manifold : Manifold
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
        input_space: str = "manifold"
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
        bias = torch.zeros((1, self.output_dim), dtype=self.params_dtype)
        self.bias = ManifoldParameter(bias, requires_grad=self.requires_grad, manifold=self.manifold)

        assert input_space in ["tangent", "manifold"], "input_space must be either 'tangent' or 'manifold'"
        self.input_space = input_space

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x of shape (B, in_dim) where the hyperbolic_axis is last
        Parameters: self.weight of shape (out_dim, in_dim), self.bias of shape (1, out_dim)
        Output: res of shape (B, out_dim)
        """
        assert self.manifold.is_in_manifold(self.bias, axis=self.hyperbolic_axis)

        if self.input_space == "manifold":
            x = self.manifold.logmap_0(x, axis=self.hyperbolic_axis)
        else:
            x, = self.manifold._2manifold_dtype([x])

        x = (x.unsqueeze(-1) * self.weight.T.unsqueeze(0)).sum(dim=1) # (B, out_dim)
        x = self.manifold.expmap_0(x, axis=self.hyperbolic_axis, backproject=self.backproject) # (B, out_dim)
        res = self.manifold.addition(x, self.bias, axis=self.hyperbolic_axis, backproject=self.backproject) # (B, out_dim)
        return res

class HyperbolicLinearPoincarePP(torch.nn.Module):
    """
    Module to compute the 'Hyperbolic Neural Networks ++' fully connected layer:
        0) Project the input tensor onto the manifold (optional)
        1) Compute the multinomial linear regression score(s)
        2) Calculate the generalized linear transformation from the reggression score(s)

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

        v = compute_mlr_PoincarePP(self.manifold, x, self.weight, self.bias, self.hyperbolic_axis,
                                   self.clamping_factor, self.smoothing_factor)
        sqrt_c = self.manifold.c.sqrt()
        w = sinh(sqrt_c * v) / sqrt_c # (B, out_dim)
        w2 = w.pow(2).sum(axis=self.hyperbolic_axis, keepdim=True) # (B, 1)
        denom = 1 + (1 + self.manifold.c * w2).sqrt() # (B, 1)
        res = w / denom # (B, out_dim)
        if self.backproject:
            res = self.manifold.proj(res, axis=self.hyperbolic_axis)
        return res
