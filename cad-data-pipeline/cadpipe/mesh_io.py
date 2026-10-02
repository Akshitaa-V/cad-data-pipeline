"""Readers for triangle mesh formats (STL binary/ASCII, OBJ) and mesh checks."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class MeshParseError(ValueError):
    """Raised when a mesh file cannot be read."""


@dataclass
class Mesh:
    vertices: np.ndarray  # (V, 3) float64, welded
    faces: np.ndarray     # (F, 3) int64

    @property
    def triangles(self) -> np.ndarray:
        return self.vertices[self.faces]


def _weld(tris: np.ndarray, decimals: int = 6) -> Mesh:
    """Merge duplicate corners so shared edges become shared indices."""
    flat = tris.reshape(-1, 3)
    keys = np.round(flat, decimals)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    return Mesh(uniq.astype(np.float64), inverse.reshape(-1, 3).astype(np.int64))


def read_stl(path: str | Path) -> Mesh:
    data = Path(path).read_bytes()
    if len(data) < 84:
        raise MeshParseError("file too short for STL")
    n = struct.unpack("<I", data[80:84])[0]
    if 84 + 50 * n == len(data):  # binary: size matches the triangle count
        rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        tris = np.frombuffer(data, dtype=rec, count=n, offset=84)["v"].astype(np.float64)
        return _weld(tris)
    text = data.decode("ascii", errors="replace")
    if not text.lstrip().lower().startswith("solid"):
        raise MeshParseError(f"binary STL size mismatch: header says {n} triangles")
    verts = [
        [float(x) for x in line.split()[1:4]]
        for line in text.splitlines()
        if line.strip().startswith("vertex")
    ]
    if not verts or len(verts) % 3:
        raise MeshParseError("ASCII STL has no complete triangles")
    return _weld(np.asarray(verts).reshape(-1, 3, 3))


def read_obj(path: str | Path) -> Mesh:
    verts, faces = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append([float(x) for x in parts[1:4]])
        elif parts[0] == "f":
            idx = [int(p.split("/")[0]) for p in parts[1:]]
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
            for k in range(1, len(idx) - 1):  # fan-triangulate polygons
                faces.append([idx[0], idx[k], idx[k + 1]])
    if not faces:
        raise MeshParseError("OBJ has no faces")
    v = np.asarray(verts, np.float64)
    f = np.asarray(faces, np.int64)
    if f.max() >= len(v):
        raise MeshParseError("OBJ face index out of range")
    return _weld(v[f])


def read_mesh(path: str | Path) -> Mesh:
    suffix = Path(path).suffix.lower()
    if suffix == ".stl":
        return read_stl(path)
    if suffix == ".obj":
        return read_obj(path)
    raise MeshParseError(f"unsupported mesh format {suffix}")


def write_obj(mesh: Mesh, path: str | Path) -> None:
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in mesh.vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in mesh.faces]
    Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def triangle_areas(mesh: Mesh) -> np.ndarray:
    t = mesh.triangles
    return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)


def signed_volume(mesh: Mesh) -> float:
    t = mesh.triangles
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


def edge_use_counts(mesh: Mesh) -> np.ndarray:
    e = np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]])
    e.sort(axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return counts


@dataclass
class MeshReport:
    n_vertices: int
    n_faces: int
    degenerate_faces: int
    watertight: bool
    euler: int
    volume: float
    area: float

    def problems(self) -> list[str]:
        out = []
        if self.n_faces == 0:
            out.append("empty mesh")
        if self.degenerate_faces:
            out.append(f"{self.degenerate_faces} degenerate triangles")
        if not self.watertight:
            out.append("not watertight (open or non-manifold edges)")
        if self.volume <= 0:
            out.append("non-positive volume (inverted or open surface)")
        return out


def check_mesh(mesh: Mesh, area_eps: float = 1e-10) -> MeshReport:
    areas = triangle_areas(mesh)
    counts = edge_use_counts(mesh)
    n_edges = len(counts)
    return MeshReport(
        n_vertices=len(mesh.vertices),
        n_faces=len(mesh.faces),
        degenerate_faces=int((areas <= area_eps).sum()),
        watertight=bool(len(counts) and np.all(counts == 2)),
        euler=int(len(mesh.vertices) - n_edges + len(mesh.faces)),
        volume=signed_volume(mesh),
        area=float(areas.sum()),
    )
