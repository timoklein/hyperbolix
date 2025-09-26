import torch

from typing import Union

from .helpers import get_torch_dtype
from ..manifolds import Hyperboloid


class HyperbolicLinearHyperboloid(torch.nn.Module):
    """
    Module to compute the 'Hyperbolic graph convolutional neural networks' fully connected layer:
        0) Project the input tensor to the tangent space (optional)
        1) Perform matrix vector multiplication in the tangent space at the origin.
        2) Map the result to the manifold.
        3) Add the manifold bias to the result by means of the parallel transport and exponential map.

    Parameters
    ----------
    manifold : Manifold
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

    References
    ----------
    Ines Chami, et al. "Hyperbolic graph convolutional neural networks."
        Advances in neural information processing systems 32 (2019).
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
        input_space: str = "manifold"
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
        # Note: We only set the space coordinates since both parameters lie in the tangent space at the Hyperboloid origin
        weight = torch.randn((output_dim-1, input_dim-1), dtype=self.params_dtype)
        self.weight = torch.nn.Parameter(weight, requires_grad=requires_grad)
        bias = torch.zeros((1, self.output_dim-1), dtype=self.params_dtype)
        self.bias = torch.nn.Parameter(bias, requires_grad=self.requires_grad)

        assert input_space in ["tangent", "manifold"], "input_space must be either 'tangent' or 'manifold'"
        self.input_space = input_space

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x of shape (B, in_dim) where the hyperbolic_axis is last
        Parameters: self.weight of shape (out_dim-1, in_dim-1), self.bias of shape (1, out_dim-1)
        Output: res of shape (B, out_dim)
        """
        assert x.shape[self.hyperbolic_axis] == self.input_dim, \
            f"self.weight lies in the tangent space at the Hyperboloid origin, i.e. its time coordinate z0 is zero and hence omitted. " \
            f"Thus, x needs to be of dimension {(x.shape[0], self.input_dim)} but is of shape {x.shape}"

        if self.input_space == "manifold":
            x = self.manifold.logmap_0(x, axis=self.hyperbolic_axis, backproject=self.backproject)
        else:
            x, = self.manifold._2manifold_dtype([x])
            assert self.manifold.is_in_tangent_space(x, self.manifold._create_origin_from_reference(x, axis=self.hyperbolic_axis), axis=self.hyperbolic_axis)

        # Matrix-Vector multiplication in the tangent space at the Hyperboloid origin
        x_rem = x.narrow(self.hyperbolic_axis, 1, x.shape[self.hyperbolic_axis]-1) # (B, in_dim - 1)
        x = (x_rem.unsqueeze(-1) * self.weight.T.unsqueeze(0)).sum(dim=1) # (B, out_dim - 1)
        # Since the result needs to lie in the tangent space at the origin we must concatenate the time coordinate back
        x = torch.cat([torch.zeros_like(x[:,:1]), x], dim=1) # (B, out_dim)
        x = self.manifold.expmap_0(x, axis=self.hyperbolic_axis, backproject=self.backproject) # (B, out_dim)

        # Bias addition
        bias = torch.cat([torch.zeros_like(self.bias[:,:1]), self.bias], dim=1) # (1, out_dim)
        pt_bias = self.manifold.ptransp_0(bias, x, axis=self.hyperbolic_axis, backproject=self.backproject) # (1, out_dim)
        res = self.manifold.expmap(pt_bias, x, axis=self.hyperbolic_axis, backproject=self.backproject) # (B, out_dim)
        return res

class HyperbolicLinearHyperboloidFHNN(torch.nn.Module):
    """
    Module to compute the 'Fully Hyperbolic Neural Networks' fully connected layer:
        0) Project the input tensor to the manifold (optional)
        1) Apply activation and dropout (optional)
        2) Compute the time coordinate of the output via a scaled sigmoid of the weight and biases transformed
           time coordinate of the input or the result of the previous step.
        3) Compute the space coordinates of the output and rescale it such that the result lies on the manifold.

    Parameters
    ----------
    manifold : Manifold
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
    init_scale : float
        Initial value for the sigmoid scale parameter (default: 2.3)
    learnable_scale : bool
        Whether the scale parameter should be learnable (default: True)
    eps : float
        Small value to ensure that the time coordinate is bigger than 1/manifold.c.sqrt() (default: 1e-5)
    activation : torch.nn.Module or None
        Activation function to apply before the linear transformation (default: None)
    dropout : float or None
        Dropout rate to apply before the activation or linear transformation (default: None)

    References
    ----------
    Weize Chen, et al. "Fully hyperbolic neural networks."
        arXiv preprint arXiv:2105.14686 (2021).
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
        init_scale: float = 2.3,
        learnable_scale: bool = True,
        eps: float = 1e-5,
        activation: Union[None, torch.nn.Module] = None,
        dropout: Union[None, float] = None
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
        self.learnable_scale = learnable_scale
        bound = 0.02
        weight = torch.empty((output_dim, input_dim), dtype=self.params_dtype).uniform_(-bound, bound)
        weight[:, 0] = 0.0 # FHNN initializes the weights as tangent vectors w.r.t. the Hyperboloid origin
        self.weight = torch.nn.Parameter(weight, requires_grad=requires_grad)
        bias = torch.zeros((1, self.output_dim), dtype=self.params_dtype)
        self.bias = torch.nn.Parameter(bias, requires_grad=self.requires_grad)
        scale = torch.tensor(init_scale, dtype=self.params_dtype)
        self.scale = torch.nn.Parameter(scale, requires_grad=self.learnable_scale)

        assert input_space in ["tangent", "manifold"], "input_space must be either 'tangent' or 'manifold'"
        self.input_space = input_space

        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout) if (isinstance(dropout, float) and dropout > 0) else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x of shape (B, in_dim) where the hyperbolic_axis is last
        Parameters: self.weight of shape (out_dim, in_dim), self.bias of shape (1, out_dim)
        Output: res of shape (B, out_dim)
        """
        assert x.shape[self.hyperbolic_axis] == self.input_dim, \
            f"x needs to be of dimension {(x.shape[0], self.input_dim)} but is of shape {x.shape}"

        if self.input_space == "tangent":
            x = self.manifold.expmap_0(x, axis=self.hyperbolic_axis, backproject=self.backproject)
        else:
            x, = self.manifold._2manifold_dtype([x])

        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = (x.unsqueeze(-1) * self.weight.T.unsqueeze(0)).sum(dim=1) + self.bias # (B, out_dim)
        x0 = x.narrow(self.hyperbolic_axis, 0, 1) # (B, 1)
        x_rem = x.narrow(self.hyperbolic_axis, 1, x.shape[self.hyperbolic_axis]-1) # (B, out_dim - 1)
        # Ensure that self.scale is positive and add self.eps to avoid numerical issues
        res0 = self.scale.exp() * x0.sigmoid() + 1 / self.manifold.c.sqrt() + self.eps # (B, 1)
        scale = (res0.pow(2) - 1 / self.manifold.c).sqrt() / x_rem.norm(p=2, dim=self.hyperbolic_axis, keepdim=True) # (B, 1)
        res_rem = scale * x_rem # (B, out_dim - 1)
        res = torch.cat([res0, res_rem], dim=self.hyperbolic_axis) # (B, out_dim)
        return res

class HyperbolicLinearHyperboloidFHCNN(torch.nn.Module):
    """
    Module to compute the 'Fully Hyperbolic Convolutional Neural Networks' fully connected layer:
        0) Project the input tensor to the manifold (optional)
        1) Apply activation (optional)
        2) a) If normalize is True, compute the time and space coordinates of the output by applying a scaled sigmoid
              of the weight and biases transformed coordinates of the input or the result of the previous step.
           b) If normalize is False, compute the weight and biases transformed space coordinates of the input or the
              result of the previous step and set the time coordinate such that the result lies on the manifold.

    Parameters
    ----------
    manifold : Manifold
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
    init_scale : float
        Initial value for the sigmoid scale parameter (default: 2.3)
    learnable_scale : bool
        Whether the scale parameter should be learnable (default: False)
    eps : float
        Small value to ensure that the time coordinate is bigger than 1/manifold.c.sqrt() (default: 1e-5)
    activation : torch.nn.Module or None
        Activation function to apply before the linear transformation (default: None)
    normalize : bool
        Whether to normalize the space coordinates before rescaling (default: False)

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
        init_scale: float = 2.3,
        learnable_scale: bool = False,
        eps: float = 1e-5,
        activation: Union[None, torch.nn.Module] = None,
        normalize: bool = False
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
        self.learnable_scale = learnable_scale
        bound = 0.02
        weight = torch.empty((output_dim, input_dim), dtype=self.params_dtype).uniform_(-bound, bound)
        self.weight = torch.nn.Parameter(weight, requires_grad=requires_grad)
        bias = torch.zeros((1, self.output_dim), dtype=self.params_dtype)
        self.bias = torch.nn.Parameter(bias, requires_grad=self.requires_grad)
        scale = torch.tensor(init_scale, dtype=self.params_dtype)
        self.scale = torch.nn.Parameter(scale, requires_grad=self.learnable_scale)

        assert input_space in ["tangent", "manifold"], "input_space must be either 'tangent' or 'manifold'"
        self.input_space = input_space

        self.activation = activation
        self.eps = eps
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x of shape (B, in_dim) where the hyperbolic_axis is last
        Parameters: self.weight of shape (out_dim, in_dim), self.bias of shape (1, out_dim)
        Output: res of shape (B, out_dim)
        """
        assert x.shape[self.hyperbolic_axis] == self.input_dim, \
            f"x needs to be of dimension {(x.shape[0], self.input_dim)} but is of shape {x.shape}"

        if self.input_space == "tangent":
            x = self.manifold.expmap_0(x, axis=self.hyperbolic_axis, backproject=self.backproject)
        else:
            x, = self.manifold._2manifold_dtype([x])

        if self.activation is not None:
            x = self.activation(x)
        x = (x.unsqueeze(-1) * self.weight.T.unsqueeze(0)).sum(dim=1) + self.bias # (B, out_dim)
        x0 = x.narrow(self.hyperbolic_axis, 0, 1) # (B, 1)
        x_rem = x.narrow(self.hyperbolic_axis, 1, x.shape[self.hyperbolic_axis]-1) # (B, out_dim - 1)
        if self.normalize:
            x_rem_norm = x_rem.norm(p=2, dim=self.hyperbolic_axis, keepdim=True)
            # Ensure that self.scale is positive and add self.eps to avoid numerical issues
            scale = self.scale.exp() * x0.sigmoid() # (B, 1)
            res0 = (scale.pow(2) + 1 / self.manifold.c + self.eps).sqrt() # (B, 1)
            res_rem = scale * x_rem / x_rem_norm # (B, out_dim - 1)
            res = torch.cat([res0, res_rem], dim=self.hyperbolic_axis) # (B, out_dim)
            # Cast vectors with small space norm to the origin
            mask = x_rem_norm <= 1e-5
            if mask.any():
                origin = self.manifold._create_origin_from_reference(res, axis=self.hyperbolic_axis)
                res = torch.where(mask, origin, res) # (B, out_dim)
        else:
            # Compute the time component from the space component and concatenate
            res0 = (x_rem.pow(2).sum(dim=self.hyperbolic_axis, keepdim=True) + 1 / self.manifold.c).sqrt() # (B, 1)
            res = torch.cat([res0, x_rem], dim=self.hyperbolic_axis) # (B, out_dim)
        return res
