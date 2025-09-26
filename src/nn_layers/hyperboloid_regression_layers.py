import torch

from .helpers import compute_mlr_Hyperboloid, get_torch_dtype
from ..manifolds import Hyperboloid


class HyperbolicRegressionHyperboloid(torch.nn.Module):
    """
    Module to compute the 'Fully Hyperbolic Convolutional Neural Networks for Computer Vision'
    multinomial linear regression score(s):
        0) Project the input tensor onto the manifold (optional)
        1) Compute the multinomial linear regression score(s)

    Parameters
    ----------
    manifold : Hyperboloid
        The Hyperboloid manifold
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
    Ahmad Bdeir, Kristian Schwethelm, and Niels Landwehr. "Fully hyperbolic convolutional neural networks for computer vision."
        arXiv preprint arXiv:2303.15919 (2023).
    """
    def __init__(
        self,
        manifold: Hyperboloid,
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
        assert isinstance(manifold, Hyperboloid), "manifold must be an instance of Hyperboloid"
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
        # weight lies in the tangent space of the Hyperboloid origin, so the time coordinate along self.hyperbolic_axis is zero
        weight = torch.randn((output_dim, input_dim-1), dtype=self.params_dtype)
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
        Parameters: self.weight of shape (out_dim, in_dim-1), self.bias of shape (out_dim, 1)
        Output: res of shape (B, out_dim)
        """
        if self.input_space == "tangent":
            x = self.manifold.expmap_0(x, axis=self.hyperbolic_axis, backproject=self.backproject)

        res = compute_mlr_Hyperboloid(self.manifold, x, self.weight, self.bias,
                                      self.hyperbolic_axis, self.clamping_factor, self.smoothing_factor)
        return res
