import math
import torch

from typing import Dict
from ..manifolds import Hyperboloid, PoincareBall
from ..utils.math_utils import smooth_clamp, asinh, cosh, sinh


# Dictionary mapping of dtype strings to torch dtypes
DTYPE_MAP: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}

def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype representation to torch dtype.

    Parameters
    ----------
    dtype_str : str
        String representation of dtype ('float32', or 'float64')

    Returns
    -------
    torch.dtype
        Corresponding torch dtype

    Raises
    ------
    ValueError
        If dtype_str is not supported
    """
    if dtype_str not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_str}. "
                         f"Supported dtypes are: {', '.join(DTYPE_MAP.keys())}")
    return DTYPE_MAP[dtype_str]

def compute_mlr_Hyperboloid(manifold: Hyperboloid, x: torch.Tensor, z: torch.Tensor, r: torch.Tensor,
                            hyperbolic_axis: int, clamping_factor: float, smoothing_factor: float) -> torch.Tensor:
        """
        Internal method for computing the 'Fully Hyperbolic Convolutional Neural Networks for Computer Vision'
        multinomial linear regression.

        Parameters
        ----------
        manifold : Hyperboloid
            The Hyperboloid manifold
        x : torch.Tensor (B, in_dim)
            Hyperboloid point(s)
        z : torch.Tensor (out_dim, in_dim-1)
            Hyperplane tangent normal(s)
        r : torch.Tensor (out_dim, 1)
            Hyperplane Hyperboloid translation(s) defined by the scalar r and z
        clamping: float
            Clamping value for the output
        smoothing: float
            Smoothing factor for the output

        Returns
        -------
        res : torch.Tensor (B, out_dim)
            The multinomial linear regression score(s) of x with respect to the linear model(s) defined by z and r.

        References
        ----------
        Ahmad Bdeir, Kristian Schwethelm, and Niels Landwehr. "Fully hyperbolic convolutional neural networks for computer vision."
            arXiv preprint arXiv:2303.15919 (2023).
        """
        assert x.shape[hyperbolic_axis] == z.shape[hyperbolic_axis] + 1, \
            f"z/self.weight lies in the tangent space at the Hyperboloid origin, i.e. its time coordinate z0 is zero and hence omitted. " \
            f"Thus, x needs to be of dimension {(x.shape[0], z.shape[hyperbolic_axis]+1)} but is of shape {x.shape}"

        x, z, r = manifold._2manifold_dtype([x, z, r])
        sqrt_c = manifold.c.sqrt()
        sqrt_cr = sqrt_c * r.T # (1, out_dim)
        z_norm = z.norm(p=2, dim=hyperbolic_axis, keepdim=True).clamp_min(manifold.min_enorm).T # (1, out_dim)
        x0 = x.narrow(hyperbolic_axis, 0, 1) # (B, 1)
        x_rem = x.narrow(hyperbolic_axis, 1, x.shape[hyperbolic_axis]-1) # (B, in_dim-1)
        zx_rem = (x_rem.unsqueeze(-1) * z.T.unsqueeze(0)).sum(dim=1) # (B, out_dim)
        alpha = -x0 * sinh(sqrt_cr) * z_norm + cosh(sqrt_cr) * zx_rem # (B, out_dim)
        asinh_arg = sqrt_c * alpha / z_norm # (B, out_dim)
        # Improve the performance by smoothly clamping the input of asinh() to approximately the range of ...
        # ... [-16*clamping_factor, 16*clamping_factor] for float32
        # ... [-36*clamping_factor, 36*clamping_factor] for float64
        eps = torch.finfo(torch.float32).eps if manifold.dtype == torch.float32 else torch.finfo(torch.float64).eps
        clamp = clamping_factor * float(math.log(2 / eps))
        asinh_arg = smooth_clamp(asinh_arg, -clamp, clamp, smoothing_factor) # (B, out_dim)
        signed_dist2hyp = asinh(asinh_arg) / sqrt_c # (B, out_dim)
        res = z_norm * signed_dist2hyp # (B, out_dim)
        return res

def compute_mlr_PoincarePP(manifold: PoincareBall, x: torch.Tensor, z: torch.Tensor, r: torch.Tensor,
                           hyperbolic_axis: int, clamping_factor: float, smoothing_factor: float) -> torch.Tensor:
        """
        Internal method for computing the 'Hyperbolic Neural Networks ++' multinomial linear regression.

        Parameters
        ----------
        manifold : PoincareBall
            The PoincareBall manifold
        x : torch.Tensor (B, in_dim)
            PoincareBall point(s)
        z : torch.Tensor (out_dim, in_dim)
            Hyperplane tangent normal(s) lying in the tangent space at the origin
        r : torch.Tensor (out_dim, 1)
            Hyperplane PoincareBall translation(s) defined by the scalar r and z
        clamping: float
            Clamping value for the output
        smoothing: float
            Smoothing factor for the output

        Returns
        -------
        res : torch.Tensor (B, out_dim)
            The multinomial linear regression score(s) of x with respect to the linear model(s) defined by z and r.

        References
        ----------
        Shimizu Ryohei, Yusuke Mukuta, and Tatsuya Harada. "Hyperbolic neural networks++."
            arXiv preprint arXiv:2006.08210 (2020).
        """
        x, z, r = manifold._2manifold_dtype([x, z, r])
        sqrt_c = manifold.c.sqrt()
        sqrt_c2r = 2 * sqrt_c * r.T # (1, out_dim)
        z_norm = z.norm(p=2, dim=hyperbolic_axis, keepdim=True).clamp_min(manifold.min_enorm) # (out_dim, 1)
        lambda_x = manifold._lambda(x, axis=hyperbolic_axis) # (B, 1)
        z_unitx = (x.unsqueeze(-1) * (z / z_norm).T).sum(dim=1) # (B, out_dim)
        asinh_arg = (1-lambda_x) * sinh(sqrt_c2r) + sqrt_c * lambda_x * cosh(sqrt_c2r) * z_unitx # (B, out_dim)
        # Improve the performance by smoothly clamping the input of asinh() to approximately the range of ...
        # ... [-16*clamping_factor, 16*clamping_factor] for float32
        # ... [-36*clamping_factor, 36*clamping_factor] for float64
        eps = torch.finfo(torch.float32).eps if manifold.dtype == torch.float32 else torch.finfo(torch.float64).eps
        clamp = clamping_factor * float(math.log(2 / eps))
        asinh_arg = smooth_clamp(asinh_arg, -clamp, clamp, smoothing_factor) # (B, out_dim)
        signed_dist2hyp = asinh(asinh_arg) / sqrt_c # (B, out_dim)
        res = 2 * z_norm.T * signed_dist2hyp # (B, out_dim)
        return res
