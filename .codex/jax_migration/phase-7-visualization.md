# Phase 7 – Visualization & HoroPCA (Optional)

## Rationale
Plotting and HoroPCA are not required for core manifold/optimizer parity. Port only if needed for demos or papers.

## TODOs
- [ ] Decide scope: keep Torch path for `vis_utils.py` and `horo_pca.py` or port selected pieces to JAX/NumPy.
- [ ] If porting, replace Torch ops with `jax.numpy` and move any training‑like loops to JAX (or keep them NumPy/scikit‑learn).
- [ ] Ensure all plotting calls convert device arrays to host (`np.asarray`) before Matplotlib.
- [ ] Add smoke tests for projections and plots (non‑graphical checks).

## Acceptance Criteria
- Visualisations run without Torch installed; optional if out of scope for the initial JAX release.
