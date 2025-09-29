#!/usr/bin/env python3
"""Export PyTorch parameters and activations for JAX parity testing."""

import sys
import os
import torch
import pickle
import json
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

# Ensure the legacy Torch packages import the same way they do in the test suite.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the modules we want to test
from src.manifolds import Euclidean, Hyperboloid, PoincareBall
from src.manifolds.manifold import Manifold, ManifoldParameter


def _tensor_to_numpy(tensor: torch.Tensor):
    """Detach, move to CPU, and convert a tensor to NumPy for serialization."""
    return tensor.detach().cpu().numpy()


def _tree_to_numpy(tree):
    """Recursively convert tensors within nested structures to NumPy arrays."""
    if isinstance(tree, torch.Tensor):
        return _tensor_to_numpy(tree)
    if isinstance(tree, dict):
        return {key: _tree_to_numpy(value) for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        converted = [_tree_to_numpy(item) for item in tree]
        return type(tree)(converted) if isinstance(tree, tuple) else converted
    return tree
from src.optim import RiemannianSGD, RiemannianAdam


def export_manifold_data():
    """Export manifold operations data for parity testing."""
    torch.manual_seed(42)

    manifold_data = {}

    # Test configurations
    configs = [
        {"manifold_cls": Euclidean, "name": "Euclidean", "dim": 5, "dtype": "float32"},
        {"manifold_cls": Euclidean, "name": "Euclidean", "dim": 5, "dtype": "float64"},
        {"manifold_cls": Hyperboloid, "name": "Hyperboloid", "dim": 5, "dtype": "float32"},
        {"manifold_cls": Hyperboloid, "name": "Hyperboloid", "dim": 5, "dtype": "float64"},
        {"manifold_cls": PoincareBall, "name": "PoincareBall", "dim": 5, "dtype": "float32"},
        {"manifold_cls": PoincareBall, "name": "PoincareBall", "dim": 5, "dtype": "float64"},
    ]

    for config in configs:
        manifold_cls = config["manifold_cls"]
        name = config["name"]
        dim = config["dim"]
        dtype = config["dtype"]

        torch_dtype = getattr(torch, dtype)

        # Create manifold with dtype-aware curvature handling
        if manifold_cls is Euclidean:
            manifold = manifold_cls(dtype=torch_dtype)
        else:
            curvature = torch.tensor([1.0], dtype=torch_dtype)
            manifold = manifold_cls(c=curvature, trainable_c=False, dtype=torch_dtype)

        key = f"{name}_{dim}_{dtype}"
        manifold_data[key] = {
            "manifold_name": name,
            "dimension": dim,
            "dtype": dtype,
            "curvature": manifold.c.item(),
        }

        # Generate test points on the manifold
        batch_size = 10
        if name == "Euclidean":
            # Points in Euclidean space
            x = torch.randn(batch_size, dim, dtype=torch_dtype)
            y = torch.randn(batch_size, dim, dtype=torch_dtype)
            tangent_vecs = torch.randn(batch_size, dim, dtype=torch_dtype)

        elif name == "Hyperboloid":
            # Points on hyperboloid (first coord positive)
            x_space = torch.randn(batch_size, dim - 1, dtype=torch_dtype)
            x_space_sq = x_space.pow(2).sum(dim=1, keepdim=True)
            x0 = torch.sqrt(torch.ones_like(x_space_sq) + x_space_sq)
            x = torch.cat([x0, x_space], dim=1)

            y_space = torch.randn(batch_size, dim - 1, dtype=torch_dtype)
            y_space_sq = y_space.pow(2).sum(dim=1, keepdim=True)
            y0 = torch.sqrt(torch.ones_like(y_space_sq) + y_space_sq)
            y = torch.cat([y0, y_space], dim=1)

            # Tangent vectors (orthogonal to x)
            tangent_space = torch.randn(batch_size, dim - 1, dtype=torch_dtype)
            tangent_0 = (x_space * tangent_space).sum(dim=1, keepdim=True) / x0
            tangent_vecs = torch.cat([tangent_0, tangent_space], dim=1)
            tangent_vecs = manifold.tangent_proj(tangent_vecs, x, axis=1)

        elif name == "PoincareBall":
            # Points in Poincare ball (norm < 1/sqrt(c))
            x = torch.randn(batch_size, dim, dtype=torch_dtype)
            x = 0.8 * x / x.norm(dim=1, keepdim=True).clamp_min(1.0)  # Ensure inside ball
            x = manifold.proj(x, axis=1)

            y = torch.randn(batch_size, dim, dtype=torch_dtype)
            y = 0.8 * y / y.norm(dim=1, keepdim=True).clamp_min(1.0)
            y = manifold.proj(y, axis=1)

            tangent_vecs = torch.randn(batch_size, dim, dtype=torch_dtype)
            tangent_vecs = manifold.tangent_proj(tangent_vecs, x, axis=1)

        # Store test inputs
        manifold_data[key]["inputs"] = {
            "x": _tensor_to_numpy(x),
            "y": _tensor_to_numpy(y),
            "tangent_vecs": _tensor_to_numpy(tangent_vecs),
            "scalars": _tensor_to_numpy(torch.tensor([0.5, 1.0, 2.0], dtype=torch_dtype)),
        }

        # Compute and store manifold operations
        with torch.no_grad():
            try:
                # Distance operations
                dist_xy = manifold.dist(x, y, axis=1)
                dist_x0 = manifold.dist_0(x, axis=1)

                # Exponential/logarithmic maps
                exp_vecs = manifold.expmap(tangent_vecs, x, axis=1, backproject=True)
                log_vecs = manifold.logmap(y, x, axis=1, backproject=True)

                # Projections
                proj_x = manifold.proj(x, axis=1)

                # Transport operations
                transported = manifold.ptransp(tangent_vecs, x, y, axis=1, backproject=True)

                # Scalar multiplication (if not Hyperboloid addition)
                scalar_mul = manifold.scalar_mul(
                    torch.tensor([2.0], dtype=torch_dtype),
                    x, axis=1, backproject=True
                )

                manifold_data[key]["outputs"] = {
                    "dist_xy": _tensor_to_numpy(dist_xy),
                    "dist_x0": _tensor_to_numpy(dist_x0),
                    "expmap": _tensor_to_numpy(exp_vecs),
                    "logmap": _tensor_to_numpy(log_vecs),
                    "proj": _tensor_to_numpy(proj_x),
                    "ptransp": _tensor_to_numpy(transported),
                    "scalar_mul": _tensor_to_numpy(scalar_mul),
                }

            except Exception as e:
                print(f"Warning: Failed to compute some operations for {key}: {e}")
                manifold_data[key]["error"] = str(e)

    return manifold_data


def export_optimizer_data():
    """Export optimizer state and updates for parity testing."""
    torch.manual_seed(42)

    optimizer_data = {}

    # Test with different manifolds
    dtype = torch.float32
    manifolds: list[Manifold] = [
        Euclidean(dtype=dtype),
        Hyperboloid(dtype=dtype),
        PoincareBall(dtype=dtype),
    ]

    for manifold in manifolds:
        manifold_name = manifold.name

        # Create test parameters
        param_shapes = [(5,), (3, 4), (2, 3, 4)]

        for i, shape in enumerate(param_shapes):
            # Generate appropriate parameters for the manifold
            param_tensor = torch.randn(shape, dtype=manifold.dtype)
            if manifold_name != "Euclidean":
                # Project to the manifold to avoid invalid initial points
                param_tensor = manifold.proj(param_tensor, axis=-1)

            param = ManifoldParameter(param_tensor.clone(), requires_grad=True, manifold=manifold)

            # Create Euclidean gradients; keep a copy for re-use across optimizers
            grad_tensor = torch.randn_like(param)
            param.grad = grad_tensor.clone()

            initial_params_tensor = param.detach().clone()
            grad_numpy = _tensor_to_numpy(grad_tensor)

            key = f"{manifold_name}_shape_{i}"
            optimizer_data[key] = {
                "manifold_name": manifold_name,
                "shape": list(shape),
                "initial_params": _tensor_to_numpy(initial_params_tensor),
                "gradients": grad_numpy,
            }

            # Test SGD optimizer
            sgd = RiemannianSGD([param], lr=0.01)
            sgd_state_before = _tree_to_numpy(sgd.state_dict())

            sgd.step()

            sgd_state_after = _tree_to_numpy(sgd.state_dict())

            optimizer_data[key]["sgd"] = {
                "lr": 0.01,
                "state_before": sgd_state_before,
                "state_after": sgd_state_after,
                "params_after": _tensor_to_numpy(param.detach()),
            }

            # Reset params for Adam test
            param.data.copy_(initial_params_tensor)
            param.grad = grad_tensor.clone()

            # Test Adam optimizer
            adam = RiemannianAdam([param], lr=0.01)
            adam_state_before = _tree_to_numpy(adam.state_dict())

            adam.step()

            adam_state_after = _tree_to_numpy(adam.state_dict())

            optimizer_data[key]["adam"] = {
                "lr": 0.01,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "state_before": adam_state_before,
                "state_after": adam_state_after,
                "params_after": _tensor_to_numpy(param.detach()),
            }

    return optimizer_data


def main():
    """Export all parity validation data."""
    print("Exporting PyTorch parity validation data...")

    # Create output directory
    output_dir = Path(__file__).parent / "baselines"
    output_dir.mkdir(exist_ok=True)

    # Export manifold data
    print("Exporting manifold operations...")
    manifold_data = export_manifold_data()

    # Export optimizer data
    print("Exporting optimizer operations...")
    optimizer_data = export_optimizer_data()

    # Create comprehensive dataset
    parity_data = {
        "export_info": {
            "timestamp": datetime.now().isoformat(),
            "torch_version": torch.__version__,
            "seed": 42,
        },
        "manifolds": manifold_data,
        "optimizers": optimizer_data,
    }

    # Save as pickle for exact numerical precision
    pickle_file = output_dir / f"pytorch_parity_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump(parity_data, f)

    # Save metadata as JSON for easy inspection
    json_file = output_dir / f"pytorch_parity_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    metadata = {
        "export_info": parity_data["export_info"],
        "manifolds": {k: {
            "manifold_name": v["manifold_name"],
            "dimension": v["dimension"],
            "dtype": v["dtype"],
            "curvature": v["curvature"],
            "input_shapes": {
                "x": list(v["inputs"]["x"].shape),
                "y": list(v["inputs"]["y"].shape),
                "tangent_vecs": list(v["inputs"]["tangent_vecs"].shape),
            } if "inputs" in v else {},
            "has_outputs": "outputs" in v,
            "error": v.get("error"),
        } for k, v in manifold_data.items()},
        "optimizers": {k: {
            "manifold_name": v["manifold_name"],
            "shape": v["shape"],
            "has_sgd": "sgd" in v,
            "has_adam": "adam" in v,
        } for k, v in optimizer_data.items()},
    }

    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported parity data to:")
    print(f"  {pickle_file}")
    print(f"  {json_file}")
    print(f"Manifold configs: {len(manifold_data)}")
    print(f"Optimizer configs: {len(optimizer_data)}")


if __name__ == "__main__":
    main()
