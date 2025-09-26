import torch

from typing import Any, Dict, Iterable, Tuple, Union
from ..manifolds import ManifoldParameter, Euclidean


__all__ = ["RiemannianAdam"]


class RiemannianAdam(torch.optim.Adam):
    """
    Riemannian Adam with the same API as :class:`torch.optim.Adam`.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize or dicts defining parameter groups
    lr : float (optional)
        Learning rate (default: 1e-3)
    betas : Tuple[float, float] (optional)
        Coefficients used for computing running averages of gradient
        and its square (default: (0.9, 0.999))
    eps : float (optional)
        Term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay : float (optional)
        Weight decay (L2 penalty) (default: 0)
    amsgrad : bool (optional)
        Whether to use the AMSGrad variant of this algorithm
        from the paper `On the Convergence of Adam and Beyond`_ (default: False)

    Other Parameters
    ----------------
    expmap_update : bool
        Update the parameters with exponential map instead of retraction (default: False)
    backproject : bool
        Whether to project results back to the manifold (default: True)
    hyperbolic_axis : int
        Axis along which the parameters are hyperbolic (default: -1)

    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    References
    ----------
    Max Kochurov, Rasul Karimov and Serge Kozlukov. "Geoopt: Riemannian Optimization in PyTorch."
        arXiv (2020).
    """
    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        expmap_update: bool = False,
        backproject: bool = True,
        hyperbolic_axis: int = -1
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, **defaults)
        self.expmap_update = expmap_update
        self.backproject = backproject
        self.hyperbolic_axis = hyperbolic_axis

    def step(self, closure=None) -> None:
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                betas = group["betas"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]
                learning_rate = group["lr"]
                amsgrad = group["amsgrad"]
                for point in group["params"]:
                    grad = point.grad
                    if grad is None:
                        continue

                    # Flag for hyperbolic parameters
                    param_is_hyperbolic = isinstance(point, ManifoldParameter) and not isinstance(point.manifold, Euclidean)
                    if param_is_hyperbolic:
                        manifold = point.manifold
                    else:
                        manifold = Euclidean()

                    state = self.state[point]
                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(point)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(point)
                        if amsgrad:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(point)
                    state["step"] += 1
                    # Make local variables for easy access
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    # Actual step
                    grad.add_(point, alpha=weight_decay)
                    grad = manifold.egrad2rgrad(grad, point, axis=self.hyperbolic_axis)
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])

                    if param_is_hyperbolic:
                        # Hyperbolic parameter: Compute <grad, grad>_x in tangent space
                        exp_avg_sq_new = manifold.tangent_inner(grad, grad, point, axis=self.hyperbolic_axis)
                        exp_avg_sq_new = exp_avg_sq_new.to(grad.dtype)
                    else:
                        # Euclidean parameter: Compute grad^2 component-wise
                        exp_avg_sq_new = grad.pow(2)

                    exp_avg_sq.mul_(betas[1]).add_(exp_avg_sq_new, alpha=1 - betas[1])
                    bias_correction1 = 1 - betas[0] ** state["step"]
                    bias_correction2 = 1 - betas[1] ** state["step"]
                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq.div(bias_correction2).sqrt_()
                    else:
                        denom = exp_avg_sq.div(bias_correction2).sqrt_()
                    # Get the direction for ascend
                    direction = exp_avg.div(bias_correction1) / denom.add_(eps)

                    if self.expmap_update:
                        # Exact update on the manifold using the exponential map
                        new_point = manifold.expmap(-learning_rate * direction, point, axis=self.hyperbolic_axis, backproject=self.backproject)
                    else:
                        # First-order approximation of the update using the retraction mapping
                        new_point = manifold.retraction(-learning_rate * direction, point, axis=self.hyperbolic_axis, backproject=self.backproject)
                    # Parallel transport the exponential averaging to the new point
                    exp_avg_new = manifold.ptransp(exp_avg, point, new_point, axis=self.hyperbolic_axis)
                    # Use copy only for user facing point
                    new_point = new_point.to(point.dtype)
                    exp_avg_new = exp_avg_new.to(exp_avg.dtype)
                    point.copy_(new_point)
                    exp_avg.copy_(exp_avg_new)
        return loss
