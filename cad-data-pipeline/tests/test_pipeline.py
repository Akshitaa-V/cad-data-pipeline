"""End-to-end tests. These need CadQuery to generate parts."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cadquery")

from cadpipe.dataset import Config, build
from cadpipe.generate import FAMILIES, build_part, export_part, sample_params
from cadpipe.loader import CadDataset
from cadpipe.mesh_io import check_mesh, read_stl
from cadpipe.step_parser import brep_graph, parse_step

SMOKE = Path(__file__).parent.parent / "configs" / "smoke.yaml"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    cfg = Config.load(SMOKE)
    cfg.out_dir = str(tmp_path_factory.mktemp("ds") / "data")
    stats = build(cfg)
    return cfg, stats, Path(cfg.out_dir)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_parser_matches_cad_kernel(family, tmp_path):
    rng = np.random.default_rng(11)
    for i in range(3):
        params = sample_params(family, rng)
        ref = export_part(build_part(family, params), tmp_path / f"p{i}", 0.01, 0.1)
        g = brep_graph(parse_step(tmp_path / ref["step"]))
        assert (g.n_faces, g.n_edges, g.n_vertices) == (
            ref["kernel_faces"], ref["kernel_edges"], ref["kernel_vertices"])
        rep = check_mesh(read_stl(tmp_path / ref["stl"]))
        assert rep.watertight
        assert rep.volume == pytest.approx(ref["kernel_volume"], rel=1e-2)


def test_hole_count_changes_topology(tmp_path):
    base = {"width": 60, "depth": 30, "thickness": 4, "hole_d": 3, "corner_r": 1}
    faces = []
    for n in (1, 2, 3):
        ref = export_part(build_part("plate_with_holes", {**base, "n_holes": n}), tmp_path / f"h{n}", 0.01, 0.1)
        faces.append(brep_graph(parse_step(tmp_path / ref["step"])).n_faces)
    assert faces[1] - faces[0] == 1 and faces[2] - faces[1] == 1


def test_every_injected_fault_is_quarantined(built):
    _, stats, _ = built
    for kind in ("truncated_stl", "broken_step", "open_mesh"):
        assert stats["caught"].get(kind, 0) == stats["injected"].get(kind, 0)
    assert stats["clean_parts_wrongly_quarantined"] == 0
    dups = stats["injected"]["duplicate"]
    assert stats["caught"]["duplicate"] + stats["duplicates_kept_as_sole_valid_copy"] == dups


def test_quarantine_records_reasons_and_moves_files(built):
    _, _, root = built
    rows = [json.loads(ln) for ln in (root / "quarantine.jsonl").read_text().splitlines()]
    assert rows and all(r["reason"] for r in rows)
    for r in rows:
        assert (root / "quarantine" / r["stl"]).exists()
        assert not (root / "raw" / r["stl"]).exists()


def test_splits_are_disjoint_and_stratified(built):
    cfg, _, root = built
    rows = [json.loads(ln) for ln in (root / "manifest.jsonl").read_text().splitlines()]
    ids = [r["part_id"] for r in rows]
    assert len(ids) == len(set(ids))
    for fam in cfg.families:
        splits = {r["split"] for r in rows if r["family"] == fam}
        assert splits == {"train", "val", "test"}


def test_no_duplicate_geometry_survives(built):
    _, _, root = built
    sigs = [json.loads(ln)["signature"] for ln in (root / "manifest.jsonl").read_text().splitlines()]
    assert len(sigs) == len(set(sigs))


def test_loader_returns_consistent_arrays(built):
    cfg, stats, root = built
    ds = CadDataset(root)
    assert len(ds) == stats["kept"]
    assert len(CadDataset(root, "train")) == stats["splits"]["train"]
    item = ds[0]
    assert item["points"].shape == (cfg.n_points, 3)
    assert np.linalg.norm(item["points"], axis=1).max() <= 1.0 + 1e-5
    assert item["voxels"].shape == (cfg.voxel_resolution,) * 3
    n_faces = item["face_features"].shape[0]
    assert item["edge_index"].max() < n_faces
    assert item["edge_features"].shape[0] == item["edge_index"].shape[1]


def test_rebuild_is_reproducible(built, tmp_path):
    _, _, root = built
    cfg2 = Config.load(SMOKE)
    cfg2.out_dir = str(tmp_path / "again")
    build(cfg2)
    a = (root / "manifest.jsonl").read_text()
    b = (Path(cfg2.out_dir) / "manifest.jsonl").read_text()
    assert a == b
    shutil.rmtree(cfg2.out_dir)


def test_config_validation(tmp_path):
    (tmp_path / "bad.yaml").write_text("n_point: 10\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.load(tmp_path / "bad.yaml")
    (tmp_path / "split.yaml").write_text("split: [0.5, 0.2, 0.2]\n")
    with pytest.raises(ValueError, match="sum to 1"):
        Config.load(tmp_path / "split.yaml")
