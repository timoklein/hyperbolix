import os
import torch
import numpy as np
import numpy.typing as npt

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Union
from .horo_pca import HoroPCA, compute_frechet_mean, center_data
from ..manifolds import Manifold, Hyperboloid, PoincareBall


def create_figure(points: torch.Tensor,
                  _manifold: Manifold,
                  labels: Union[npt.ArrayLike, None]=None,
                  edges: Union[Tuple[List[int], List[int]], None]=None,
                  hyperplanes: Union[Tuple[torch.Tensor, torch.Tensor], None]=None,
                  settings: Dict[str, Union[str, bool]]=None
                  ) -> None:
    """
    Create and save a 2D visualization of hyperbolic points, geodesics, and hyperplanes in the PoincareBall.

    Parameters
    ----------
    points : torch.Tensor
        Manifold point(s) to be visualized
    manifold_ : Manifold
        The manifold type (e.g. Hyperboloid, PoincareBall)
    labels : Union[npt.ArrayLike, None] (optional)
        Labels for the manifold point(s) (default: None)
    edges : Tuple[List[int], List[int]] (optional)
        Index pairs for which edges/geodesics should be visualized.
        The Indices refer to the input 'points' (default: None)
    hyperplanes : Tuple[torch.Tensor, torch.Tensor] (optional)
        Tuple containing hyperplane normals and base points (default: None)
    settings : Dict[str, Union[str, bool]] (optional)
        Dictionary of settings for the visualization.
        If not provided, the following default settings are used:
        - "plot_manifold_dtype": torch.float64 (Data type for plotting)
        - "dim_red_method": "HoroPCA" {"HoroPCA", "tangent PCA"} (Dimensionality reduction method)
        - "title": "Hyperbolic Embeddings" (Title of the plot)
        - "show_origin": True (Whether to show the origin)
        - "save_figure": False (Whether to save the figure or return it)
        - "file_name": "hyperbolic_embeddings" (Name of the saved file)
        - "file_path": None (Path to save the file, current directory if None)
        - "file_format": "png" (Format of the saved file)
    """
    default_settings = {
        "plot_manifold_dtype": torch.float64,
        "dim_red_method": "HoroPCA",
        "title": "Hyperbolic Embeddings",
        "show_origin": True,
        "save_figure": False,
        "file_name": "hyperbolic_embeddings",
        "file_path": None,
        "file_format": "png",
    }

    if settings is not None:
        default_settings.update(settings)
    settings = default_settings

    # Create the PoincareBall with curvature of the same type as 'plot_manifold_dtype'
    # Default "plot_manifold_dtype" is double precision to avoid representational instabilities
    poincare = PoincareBall(c=_manifold.c.to(settings['plot_manifold_dtype']), dtype=settings['plot_manifold_dtype'])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')
    poincare_closure = 1 / poincare.c.sqrt().cpu().detach()
    ax.set_xlim(-1.1 * poincare_closure, 1.1 * poincare_closure)
    ax.set_ylim(-1.1 * poincare_closure, 1.1 * poincare_closure)

    # Draw the closure of the PoincareBall
    circle = plt.Circle((0, 0), poincare_closure, fill=False, color='black', zorder=5)
    ax.add_artist(circle)

    if settings['show_origin']:
        ax.scatter(0, 0, marker='x', c='black', s=50, zorder=5)

    # Project points onto the 2d PoincareBall
    points = points.detach()
    assert _manifold.is_in_manifold(points), "Points are not in the manifold"
    if isinstance(_manifold, PoincareBall) and points.shape[-1] > 2:
        points = poincare.to_hyperboloid(points)
        if hyperplanes is not None:
            hyperplanes = (poincare.to_hyperboloid(hyperplanes[0]),
                           poincare.to_hyperboloid(hyperplanes[1]))
        points, hyperplanes = pointsTo2dPoincare(points, poincare, hyperplanes, settings)
        ax.set_title(f"{settings['title']} ({settings['dim_red_method']})")
    elif isinstance(_manifold, Hyperboloid) and points.shape[-1] > 3:
        points, hyperplanes = pointsTo2dPoincare(points, poincare, hyperplanes, settings)
        ax.set_title(f"{settings['title']} ({settings['dim_red_method']})")
    else:
        hyperplanes = (hyperplanes[0], hyperplanes[1]) if hyperplanes is not None else None
        ax.set_title(f"{settings['title']}")

    # Plot the points in 2D
    handles = plot_2d_points(points, ax, labels)

    # Plot geodesics between specified points
    if edges is not None:
        plot_edges(points, edges, poincare, ax, handles)

    # Plot hyperplanes
    if hyperplanes is not None:
        plot_hyperplane(hyperplanes, poincare, ax, handles)

    if handles:
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    if settings['save_figure']:
        save_figure(fig, settings['file_name'], file_path=settings['file_path'], format=settings['file_format'])
    else:
        return fig

def to_hyperboloid_vis(x: torch.Tensor, poincare: PoincareBall) -> torch.Tensor:
    """Project PoincareBall points to the Hyperboloid. Before projecting to the Hyperboloid,
       we rescale the points to match the representational limitations between the PoincareBall
       and the Hyperboloid."""
    res = poincare.to_hyperboloid(x)
    return res

def pointsTo2dPoincare(x: torch.Tensor, poincare: PoincareBall,
                       hyperplanes: Union[Tuple[torch.Tensor, torch.Tensor], None]=None,
                       settings: Union[dict, None]=None) -> Tuple[npt.ArrayLike, Union[npt.ArrayLike, None]]:
    """Project Hyperboloid points and Hyperboloid hyperplanes to the 2d PoincareBall using the specified method."""
    hyperboloid = Hyperboloid(c=poincare.c, dtype=poincare.dtype)

    if hyperplanes is not None:
        hyperplane_size = hyperplanes[0].shape[0]
        hyperplanes = torch.cat((hyperplanes[0], hyperplanes[1]), dim=0)

    if settings['dim_red_method'] == 'HoroPCA':
        model = HoroPCA(n_components=2, n_in_features=x.shape[1], manifold=hyperboloid)
        model.fit(x)
        x = model.transform(x).detach()
        if hyperplanes is not None:
            hyperplanes = model.transform(hyperplanes).detach()
    elif settings['dim_red_method'] == 'tangent PCA':
        mean = compute_frechet_mean(x, hyperboloid)
        x = center_data(x, mean, hyperboloid)
        x = hyperboloid.to_poincare(x)
        x_tangent = poincare.logmap_0(x).cpu()
        model = PCA(n_components=2).fit(x_tangent)
        x_tangent = torch.from_numpy(model.transform(x_tangent))
        x = poincare.expmap_0(x_tangent.to(poincare.c.device))
        if hyperplanes is not None:
            hyperplanes = center_data(hyperplanes, mean, hyperboloid)
            hyperplanes = hyperboloid.to_poincare(hyperplanes)
            hyperplanes_tangent = poincare.logmap_0(hyperplanes).cpu()
            hyperplanes_tangent = torch.from_numpy(model.transform(hyperplanes_tangent))
            hyperplanes = poincare.expmap_0(hyperplanes_tangent.to(poincare.c.device))
    else:
        raise ValueError(f"Unknown dimensionality reduction method {settings['dim_red_method']}")

    if hyperplanes is not None:
        hyperplanes = (hyperplanes[:hyperplane_size], hyperplanes[hyperplane_size:])
    return x, hyperplanes

def plot_2d_points(x: torch.Tensor, ax: plt.Axes, labels: Union[npt.ArrayLike, None]=None) -> List[plt.Line2D]:
    """Plot 2d PoincareBall points with labels (optional)."""
    x = x.cpu()
    if labels is None:
        ax.scatter(x[:, 0], x[:, 1], c='blue', alpha=0.6, zorder=2)
        return []
    else:
        assert x.shape[0] == len(labels), "Number of labels must match number of points"
        unique_labels = np.unique(labels)
        cmap = plt.cm.tab10 if len(unique_labels) <= 10 else plt.cm.tab20
        colors = cmap(np.linspace(0, 1, len(unique_labels)))
        color_map = dict(zip(unique_labels, colors))
        ax.scatter(x[:, 0], x[:, 1], c=[color_map[label] for label in labels], alpha=0.6, zorder=2)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[label],
                              markersize=10, alpha=0.6, label=f"{label}")
                   for label in unique_labels]
        return handles

def plot_edges(points: torch.Tensor, edges: Tuple[List[int], List[int]],
               poincare: PoincareBall, ax: plt.Axes, handles: List[plt.Line2D]) -> None:
    """Plot geodesic segment(s) connecting x and y."""
    assert len(edges[0]) == len(edges[0]), "Start and end points must have the same shape"
    spacing = 100
    t = torch.linspace(0, 1.0, spacing, device=poincare.c.device).reshape(-1, 1)
    x = points[edges[0]]
    y = points[edges[1]]
    dir = poincare.addition(-x, y)
    for _x, _dir in zip(x, dir):
        # Compute points on the geodesic segment connecting x and y
        second_term = poincare.scalar_mul(t, _dir.repeat(spacing, 1), backproject=True)
        geodesic = poincare.addition(_x.repeat(spacing, 1), second_term, backproject=True)
        geodesic = geodesic.cpu()
        ax.plot(geodesic[:, 0], geodesic[:, 1], c='dimgrey', zorder=3, linewidth=1.5)
    handles.append(plt.Line2D([0], [0], color='dimgrey', label='Geodesic'))

def plot_hyperplane(hyperplanes: Tuple[torch.Tensor, torch.Tensor], poincare: PoincareBall,
                    ax: plt.Axes, handles: List[plt.Line2D]) -> None:
    """Plot hyperplane(s) and their base point(s)."""
    hyperplane_normals, hyperplane_base_points = hyperplanes
    hyperplane_normals = hyperplane_normals.reshape(-1, 2)
    hyperplane_base_points = hyperplane_base_points.reshape(-1, 2)
    hyperplane_normals[:, 0] = -hyperplane_normals[:, 0]
    hyperplane_dirs = hyperplane_normals[:, ::-1]
    spacing = 10_000
    t = torch.linspace(-500, 500, spacing, device=poincare.c.device)
    for _base, _dir in zip(hyperplane_base_points, hyperplane_dirs):
        points = poincare.expmap(torch.outer(t, _dir), _base.repeat(spacing, 1)).cpu()
        ax.plot(points[:, 0], points[:, 1], c='green', alpha=0.6, zorder=3)
    ax.scatter(hyperplane_base_points[:, 0], hyperplane_base_points[:, 1], c='green', marker='P', s=50, zorder=4)
    handles.append(plt.Line2D([0], [0], color='green', label='Hyperplane'))
    handles.append(plt.Line2D([0], [0], color='green', marker='P', linestyle='',
                              markersize=10, label='Hyperplane Base Points'))

def save_figure(fig: plt.Figure, file_name: str, file_path: Union[str, None]=None, format: str='png') -> None:
    """Save the figure to a file."""
    if file_path is None:
        file_path = os.getcwd()
    path = os.path.join(file_path, 'images')
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f"{file_name}.{format}"))
    plt.close(fig)
