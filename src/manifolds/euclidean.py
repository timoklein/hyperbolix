import torch

from typing import List
from .manifold import Manifold


class Euclidean(Manifold):
    """
    Euclidean manifold class.
    """
    def __init__(
        self,
        c: torch.Tensor = torch.tensor([0.]),
        trainable_c: bool = False,
        dtype: str | torch.dtype = "float32",
    ):
        super().__init__(torch.tensor([0.]), trainable_c=False)
        self.name = "Euclidean"
        self.dtype = dtype
        if trainable_c:
            print("Warning: trainable_c is not supported for Euclidean manifold. Setting it to False.")
        elif not torch.allclose(c, torch.zeros_like(c)):
            print("Warning: c!=0 is not supported for Euclidean manifold. Setting it to 0.")
        if dtype == "float32" or dtype == torch.float32:
            self.dtype = torch.float32
        elif dtype == "float64" or dtype == torch.float64:
            self.dtype = torch.float64
        else:
            raise ValueError(f"Unsupported dtype: {dtype}. Supported dtypes are float32, and float64.")

    def _2manifold_dtype(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Convert the list of tensor(s) xs to the Euclidean manifold's dtype.

        Parameters
        ----------
        xs : List[torch.Tensor]
            List of tensor(s)

        Returns
        -------
        res : List[torch.Tensor]
            The list of tensor(s) converted to the Euclidean manifold's dtype
        """
        res = []
        for x in xs:
            res.append(x.to(self.dtype))
        return res

    def addition(self, x: torch.Tensor, y: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Add Euclidean manifold point(s) y to Euclidean manifold point(s) x.

        Parameters
        ----------
        x : torch.Tensor
            Euclidean manifold point(s)
        y : torch.Tensor
            Euclidean manifold point(s)
        axis : int (ignored)
            Axis along which to compute the addition (default: -1)
        backproject : bool (ignored)
            Whether to project results back to the Euclidean manifold (default: True)
        Note: Axis and backproject are not used in the Euclidean manifold, but included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The sum(s) of x and y
        """
        x, y = self._2manifold_dtype([x, y])
        res = x + y
        return res

    def scalar_mul(self, r: torch.Tensor, x: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Multiply Euclidean manifold point(s) x with scalar(s) r.

        Parameters
        ----------
        r : torch.Tensor
            Scalar factor(s)
        x : torch.Tensor
            Euclidean manifold point(s)
        axis : int (ignored)
            Axis along which to compute the multiplication (default: -1)
        backproject : bool (ignored)
            Whether to project results back to the Euclidean manifold (default: True)
        Note: Axis and backproject are not used in the Euclidean manifold, but included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The product(s) of r and x
        """
        x, r = self._2manifold_dtype([x, r])
        res = r * x
        return res

    def dist(self, x: torch.Tensor, y: torch.Tensor, axis: int=-1, backproject: bool=True, version: str="default") -> torch.Tensor:
        """
        Compute the geodesic distance(s) between Euclidean manifold points x and y.

        Parameters
        ----------
        x : torch.Tensor
            Euclidean manifold point(s)
        y : torch.Tensor
            Euclidean manifold point(s)
        axis : int
            Axis along which to compute the geodesic distance (default: -1)
        backproject : bool (ignored)
            Whether to project results back to the Euclidean manifold (default: True)
        version : str (ignored)
            Version of the geodesic distance to compute (default: "default")
        Note: Backproject and version are not used in the Euclidean manifold, but included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The geodesic distance(s) between x and y
        """
        x, y = self._2manifold_dtype([x, y])
        res = (x - y).norm(p=2, dim=axis, keepdim=True)
        return res

    def dist_0(self, x: torch.Tensor, axis: int=-1, version: str="default") -> torch.Tensor:
        """
        Compute the geodesic distance(s) of Euclidean manifold point(s) x from/to the Euclidean origin.

        Parameters
        ----------
        x : torch.Tensor
            Euclidean manifold point(s)
        axis : int
            Axis along which to compute the geodesic distance (default: -1)
        version : str (ignored)
            Version of the geodesic distance to compute (default: "default")
        Note: Version is not used in the Euclidean manifold, but included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The geodesic distance(s) of x from/to the Euclidean origin
        """
        x, = self._2manifold_dtype([x])
        res = x.norm(p=2, dim=axis, keepdim=True)
        return res

    def expmap(self, v: torch.Tensor, x: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Map tangent vector(s) v at Euclidean manifold point(s) x to the Euclidean manifold.
        [Exponential map]

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            Euclidean manifold point(s)
        axis : int (ignored)
            Axis along which to compute the exponential map (default: -1)
        backproject : bool (ignored)
            Whether to project results back to the Euclidean manifold (default: True)
        Note: Axis and backproject are not used in the Euclidean manifold, but included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The point(s) after mapping v to the Euclidean manifold
        """
        v, x = self._2manifold_dtype([v, x])
        res = x + v
        return res

    def expmap_0(self, v: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Map tangent vector(s) v at the Euclidean origin to the Euclidean manifold.
        [Exponential map]

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space of the Euclidean origin
        axis : int (ignored)
            Axis along which to compute the exponential map (default: -1)
        backproject : bool (ignored)
            Whether to project results back to the Euclidean manifold (default: True)
        Note: Axis and backproject are not used in the Euclidean manifold, but included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The point(s) after mapping v to the Euclidean manifold
        """
        v, = self._2manifold_dtype([v])
        res = v
        return res

    def retraction(self, v: torch.Tensor, x: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        First-order approximation of the exponential map for vector(s) v at Euclidean manifold point(s) x.
        [Retraction map]

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            Euclidean manifold point(s)
        axis : int (ignored)
            Axis along which to compute the retraction (default: -1)
        backproject : bool (ignored)
            Whether to project results back to the Euclidean manifold (default: True)
        Note: Axis and backproject are not used in the Euclidean manifold, but included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The point(s) after approximately mapping v to the Euclidean manifold
        """
        v, x = self._2manifold_dtype([v, x])
        res = x + v
        return res

    def logmap(self, y: torch.Tensor, x: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Map Euclidean manifold point(s) y to the tangent space(s) of Euclidean manifold point(s) x.
        [Logarithmic map]

        Parameters
        ----------
        y : torch.Tensor
            Euclidean manifold point(s)
        x : torch.Tensor
            Euclidean manifold point(s)
        axis : int (ignored)
            Axis along which to compute the logarithmic map (default: -1)
        backproject : bool (ignored)
            Whether to project results back to the tangent space(s) of x (default: True)
        Note: Axis and backproject are not used in the Euclidean manifold, but included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The resulting tangent vector(s) after mapping y to the tangent space(s) of x
        """
        y, x = self._2manifold_dtype([y, x])
        res = y - x
        return res

    def logmap_0(self, y: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Map Euclidean manifold point(s) y to the tangent space of the Euclidean origin.
        [Logarithmic map]

        Parameters
        ----------
        y : torch.Tensor
            Euclidean manifold point(s)
        axis : int (ignored)
            Axis along which to compute the logarithmic map (default: -1)
        backproject : bool (ignored)
            Whether to project results back to the tangent space of the Euclidean origin (default: True)
        Note: Axis and backproject are not used in the Euclidean manifold, but included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The resulting tangent vector(s) after mapping y to the tangent space of the origin
        """
        y, = self._2manifold_dtype([y])
        res = y
        return res

    def ptransp(self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Parallel transport tangent vector(s) v from the tangent space(s) of Euclidean
        manifold point(s) x to the tangent space(s) of Euclidean manifold point(s) y.

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            Euclidean manifold point(s)
        y : torch.Tensor
            Euclidean manifold point(s)
        axis : int (ignored)
            Axis along which to compute the parallel transport (default: -1)
        backproject : bool (ignored)
            Whether to project results back to the tangent space(s) of y (default: True)
        Note: Axis and backproject are not used in the Euclidean manifold, but included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The parallel transported tangent vector(s)
        """
        v, = self._2manifold_dtype([v])
        res = v
        return res

    def ptransp_0(self, v: torch.Tensor, y: torch.Tensor, axis: int=-1, backproject: bool=True) -> torch.Tensor:
        """
        Parallel transport tangent vector(s) v from the tangent space of
        the Euclidean origin to the tangent space(s) of Euclidean manifold point(s) y.

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space of the Euclidean origin
        y : torch.Tensor
            Euclidean manifold point(s)
        axis : int (ignored)
            Axis along which to compute the parallel transport (default: -1)
        backproject : bool (ignored)
            Whether to project results back to the tangent space(s) of y (default: True)
        Note: Axis and backproject are not used in the Euclidean manifold, but included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The parallel transported tangent vector(s)
        """
        v, = self._2manifold_dtype([v])
        res = v
        return res

    def tangent_inner(self, u: torch.Tensor, v: torch.Tensor, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Compute the inner product(s) between tangent vectors u and v of the tangent space(s)
        at Euclidean manifold point(s) x with respect to the Riemannian metric of the Euclidean manifold.

        Parameters
        ----------
        u : torch.Tensor
            Vector(s) in the tangent space(s) of x
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            Euclidean manifold point(s)
        axis : int
            Axis along which to compute the tangent inner product (default: -1)

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The tangent inner product(s) of u and v
        """
        u, v = self._2manifold_dtype([u, v])
        res = (u * v).sum(dim=axis, keepdim=True)
        return res

    def tangent_norm(self, v: torch.Tensor, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Compute the norm(s) of tangent vector(s) v of the tangent space(s) at Euclidean manifold point(s) x
        with respect to the Riemannian metric of the Euclidean manifold.

        Parameters
        ----------
        v : torch.Tensor
            Vector(s) in the tangent space(s) of x
        x : torch.Tensor
            Euclidean manifold point(s)
        axis : int
            Axis along which to compute the tangent norm (default: -1)

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The tangent norm(s) of v
        """
        v, = self._2manifold_dtype([v])
        res = v.norm(p=2, dim=axis, keepdim=True)
        return res

    def egrad2rgrad(self, grad: torch.Tensor, x: torch.Tensor, axis: int=-1) -> torch.Tensor:
        """
        Compute the Riemannian gradient(s) at Euclidean manifold point(s) x from the Euclidean gradient(s).

        Parameters
        ----------
        grad : torch.Tensor
            Euclidean gradient(s)
        x : torch.Tensor
            Euclidean manifold point(s)
        axis : int (ignored)
            Axis along which to compute the Riemannian gradient (default: -1)
        Note: Axis is not used in the Euclidean manifold, but included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor
            The Riemannian gradient(s) at x
        """
        res = grad
        return res

    def proj(self, x: torch.Tensor, axis: int=-1):
        """
        Project point(s) x onto the Euclidean manifold.

        Parameters
        ----------
        x : torch.Tensor
            Point(s)
        axis : int (ignored)
            Axis along which to compute the projection (default: -1)
        Note: Axis is not used in the Euclidean manifold, but included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The projected Euclidean manifold point(s)
        """
        x, = self._2manifold_dtype([x])
        res = x
        return res

    def tangent_proj(self, v: torch.Tensor, x: torch.Tensor, axis: int=-1):
        """
        Project point(s) v onto the tangent space(s) of Euclidean manifold point(s) x.

        Parameters
        ----------
        v : torch.Tensor
            Point(s)
        x : torch.Tensor
            Euclidean manifold point(s)
        axis : int (ignored)
            Axis along which to compute the projection (default: -1)
        Note: Axis is not used in the Euclidean manifold, but included for consistency with other manifolds.

        Returns
        -------
        res : torch.Tensor (dtype=self.dtype)
            The projected tangent vector(s)
        """
        v, = self._2manifold_dtype([v])
        res = v
        return res

    def is_in_manifold(self, x: torch.Tensor, axis: int=-1) -> bool:
        """
        Check if point(s) x lie in the Euclidean manifold.

        Parameters
        ----------
        x : torch.Tensor
            Euclidean manifold point(s)
        axis : int (ignored)
            Axis along which to check if x lies in the Euclidean manifold (default: -1)
        Note: Axis is not used in the Euclidean manifold, but included for consistency with other manifolds.

        Returns
        -------
        res : bool
            True if all points x lie in the Euclidean manifold, False otherwise
        """
        res = True
        return res

    def is_in_tangent_space(self, v: torch.Tensor, x: torch.Tensor, axis: int=-1) -> bool:
        """
        Check if vector(s) v belong to the tangent space(s) at Euclidean manifold point(s) x.

        Parameters
        ----------
        v : torch.Tensor
            Vector(s)
        x : torch.Tensor
            Euclidean manifold point(s)
        axis : int (ignored)
            Axis along which to check if v belongs to the tangent space (default: -1)
        Note: Axis is not used in the Euclidean manifold, but included for consistency with other manifolds.

        Returns
        -------
        res : bool
            True if all vectors v belong to their tangent spaces, False otherwise
        """
        res = True
        return res
