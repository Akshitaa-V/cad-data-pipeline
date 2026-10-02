"""Turn meshes into ML-ready point clouds and voxel grids."""
from __future__ import annotations

import numpy as np

from .mesh_io import Mesh, triangle_areas


def sample_surface(mesh: Mesh, n_points: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted uniform sampling on the surface.

    Returns points (N, 3) and unit outward normals (N, 3) as float32. Each
    triangle is picked with probability proportional to its area, then a point
    is drawn uniformly inside it with the square-root barycentric trick.
    """
    rng = np.random.default_rng(seed)
    areas = triangle_areas(mesh)
    total = areas.sum()
    if total <= 0:
        raise ValueError("mesh has zero surface area")
    tri_idx = rng.choice(len(areas), size=n_points, p=areas / total)
    t = mesh.triangles[tri_idx]
    r1 = np.sqrt(rng.random(n_points))[:, None]
    r2 = rng.random(n_points)[:, None]
    pts = (1 - r1) * t[:, 0] + r1 * (1 - r2) * t[:, 1] + r1 * r2 * t[:, 2]
    nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return pts.astype(np.float32), nrm.astype(np.float32)


def normalize_unit_sphere(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre on the bounding-box centre and scale into the unit sphere.

    Returns the normalised points plus the centre and scale, so that the
    original coordinates can be recovered: original = p * scale + centre.
    """
    centre = (points.min(0) + points.max(0)) / 2.0
    shifted = points - centre
    scale = float(np.linalg.norm(shifted, axis=1).max()) or 1.0
    return (shifted / scale).astype(np.float32), centre.astype(np.float32), scale


def voxelize_points(points: np.ndarray, resolution: int = 32) -> np.ndarray:
    """Occupancy grid (R, R, R) of points already inside the unit sphere."""
    idx = np.clip(((points + 1.0) / 2.0 * resolution).astype(int), 0, resolution - 1)
    grid = np.zeros((resolution,) * 3, dtype=np.uint8)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    return grid


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance between two point sets."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(d.min(1).mean() + d.min(0).mean())
