import struct

import numpy as np
import pytest

from cadpipe.mesh_io import Mesh, MeshParseError, check_mesh, read_mesh, read_obj, read_stl, write_obj
from cadpipe.sampling import chamfer_distance, normalize_unit_sphere, sample_surface, voxelize_points

# Unit cube [0,1]^3 with outward-facing triangles
V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
F = np.array([
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
    [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
])
CUBE = Mesh(V, F)


def _write_binary_stl(path, tris):
    body = b"".join(struct.pack("<12fH", 0, 0, 0, *t.ravel(), 0) for t in tris)
    path.write_bytes(b"\0" * 80 + struct.pack("<I", len(tris)) + body)


def _write_ascii_stl(path, tris):
    lines = ["solid cube"]
    for t in tris:
        lines += ["facet normal 0 0 0", "outer loop"] + [f"vertex {x} {y} {z}" for x, y, z in t]
        lines += ["endloop", "endfacet"]
    path.write_text("\n".join(lines + ["endsolid cube"]))


def test_cube_checks():
    rep = check_mesh(CUBE)
    assert rep.watertight and rep.degenerate_faces == 0
    assert rep.volume == pytest.approx(1.0) and rep.area == pytest.approx(6.0)
    assert rep.euler == 2 and rep.problems() == []


def test_binary_stl_welds_shared_corners(tmp_path):
    _write_binary_stl(tmp_path / "c.stl", CUBE.triangles)
    m = read_stl(tmp_path / "c.stl")
    assert len(m.vertices) == 8 and len(m.faces) == 12
    assert check_mesh(m).watertight


def test_ascii_stl(tmp_path):
    _write_ascii_stl(tmp_path / "c.stl", CUBE.triangles)
    assert check_mesh(read_mesh(tmp_path / "c.stl")).volume == pytest.approx(1.0)


def test_truncated_binary_stl_is_rejected(tmp_path):
    _write_binary_stl(tmp_path / "c.stl", CUBE.triangles)
    data = (tmp_path / "c.stl").read_bytes()
    (tmp_path / "c.stl").write_bytes(data[:-30])
    with pytest.raises(MeshParseError, match="size mismatch"):
        read_stl(tmp_path / "c.stl")


def test_obj_roundtrip_and_quad_triangulation(tmp_path):
    write_obj(CUBE, tmp_path / "c.obj")
    assert check_mesh(read_obj(tmp_path / "c.obj")).volume == pytest.approx(1.0)
    (tmp_path / "q.obj").write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    assert len(read_obj(tmp_path / "q.obj").faces) == 2


def test_unsupported_format(tmp_path):
    (tmp_path / "x.ply").write_text("ply")
    with pytest.raises(MeshParseError):
        read_mesh(tmp_path / "x.ply")


def test_open_mesh_detected():
    rep = check_mesh(Mesh(V, F[:-1]))
    assert not rep.watertight
    assert any("watertight" in p for p in rep.problems())


def test_degenerate_triangle_detected():
    m = Mesh(np.vstack([V, [[2, 2, 2]]]), np.vstack([F, [[8, 8, 6]]]))
    assert check_mesh(m).degenerate_faces == 1


def test_inverted_mesh_has_negative_volume():
    rep = check_mesh(Mesh(V, F[:, ::-1]))
    assert rep.volume == pytest.approx(-1.0)
    assert any("volume" in p for p in rep.problems())


def test_samples_lie_on_the_surface_with_unit_normals():
    pts, nrm = sample_surface(CUBE, 2000, seed=0)
    assert pts.shape == (2000, 3) and pts.dtype == np.float32
    on_face = np.isclose(pts, 0, atol=1e-6) | np.isclose(pts, 1, atol=1e-6)
    assert np.all(on_face.any(axis=1))
    np.testing.assert_allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-5)


def test_sampling_is_area_weighted():
    # two triangles, the second with three times the area of the first
    m = Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [3, 0, 1], [0, 1, 1]], float),
             np.array([[0, 1, 2], [3, 4, 5]]))
    pts, _ = sample_surface(m, 20000, seed=1)
    share_upper = (pts[:, 2] > 0.5).mean()
    assert share_upper == pytest.approx(0.75, abs=0.02)


def test_sampling_is_deterministic():
    a, _ = sample_surface(CUBE, 100, seed=3)
    b, _ = sample_surface(CUBE, 100, seed=3)
    c, _ = sample_surface(CUBE, 100, seed=4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_normalisation_is_reversible():
    pts, _ = sample_surface(CUBE, 500, seed=0)
    pts = pts * 40 + 7
    p, centre, scale = normalize_unit_sphere(pts)
    assert np.linalg.norm(p, axis=1).max() == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(p * scale + centre, pts, atol=1e-3)


def test_voxels_and_chamfer():
    p, _, _ = normalize_unit_sphere(sample_surface(CUBE, 4000, seed=0)[0])
    grid = voxelize_points(p, 8)
    assert grid.shape == (8, 8, 8) and grid[4, 4, 4] == 0 and grid.sum() > 0  # hollow surface
    assert chamfer_distance(p[:200], p[:200]) == 0.0
