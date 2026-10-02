"""Dataset build: generate -> inject faults -> parse -> validate -> quarantine
-> deduplicate -> split -> write ML-ready shards, a manifest and a report.

Every step is driven by one YAML config, so a rebuild with the same config and
seed produces the same dataset.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import shutil
import struct
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .mesh_io import check_mesh, read_mesh
from .sampling import normalize_unit_sphere, sample_surface, voxelize_points
from .step_parser import brep_graph, length_unit_scale, parse_step

FAULT_KINDS = ("truncated_stl", "broken_step", "open_mesh")


@dataclass
class Config:
    out_dir: str = "data"
    seed: int = 42
    parts_per_family: int = 100
    families: tuple = ("plate_with_holes", "l_bracket", "flanged_shaft", "stepped_block")
    n_points: int = 2048
    voxel_resolution: int = 32
    stl_tolerance: float = 0.01
    angular_tolerance: float = 0.1
    duplicate_fraction: float = 0.05
    fault_fraction: float = 0.05
    volume_rel_tol: float = 0.01
    split: tuple = (0.8, 0.1, 0.1)
    workers: int = 4

    @classmethod
    def load(cls, path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        for k in ("families", "split"):
            if k in raw:
                raw[k] = tuple(raw[k])
        cfg = cls(**raw)
        if abs(sum(cfg.split) - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1")
        return cfg


# --------------------------------------------------------------------------
# Stage 1: generation (runs in worker processes, needs CadQuery)
# --------------------------------------------------------------------------

def _generate_one(job: dict) -> dict:
    from .generate import build_part, export_part

    stem = Path(job["raw_dir"]) / job["part_id"]
    shape = build_part(job["family"], job["params"])
    ref = export_part(shape, stem, job["stl_tolerance"], job["angular_tolerance"])
    return {**job, **ref}


def plan_jobs(cfg: Config, raw_dir: Path) -> list[dict]:
    from .generate import sample_params

    rng = np.random.default_rng(cfg.seed)
    jobs = []
    for fam in cfg.families:
        for i in range(cfg.parts_per_family):
            jobs.append({"part_id": f"{fam}_{i:04d}", "family": fam, "params": sample_params(fam, rng)})
    # Duplicates: re-emit existing parameter sets under new ids, as a scraped
    # dataset would contain the same part uploaded twice.
    n_dup = int(round(cfg.duplicate_fraction * len(jobs)))
    for k, src in enumerate(rng.choice(len(jobs), size=n_dup, replace=False)):
        j = jobs[src]
        jobs.append({"part_id": f"{j['family']}_dup{k:03d}", "family": j["family"],
                     "params": dict(j["params"]), "injected": "duplicate"})
    for j in jobs:
        j.update(raw_dir=str(raw_dir), stl_tolerance=cfg.stl_tolerance,
                 angular_tolerance=cfg.angular_tolerance)
    return jobs


# --------------------------------------------------------------------------
# Stage 2: fault injection (known ground truth for the validators)
# --------------------------------------------------------------------------

def inject_faults(records: list[dict], cfg: Config, raw_dir: Path) -> None:
    rng = np.random.default_rng(cfg.seed + 1)
    clean = [r for r in records if "injected" not in r]
    n = int(round(cfg.fault_fraction * len(clean)))
    for k, idx in enumerate(rng.choice(len(clean), size=n, replace=False)):
        rec, kind = clean[idx], FAULT_KINDS[k % len(FAULT_KINDS)]
        rec["injected"] = kind
        stl, step = raw_dir / rec["stl"], raw_dir / rec["step"]
        if kind == "truncated_stl":  # interrupted download
            data = stl.read_bytes()
            stl.write_bytes(data[: len(data) // 2])
        elif kind == "broken_step":  # faces removed by a faulty exporter
            lines = [ln for ln in step.read_text(encoding="latin-1").splitlines()
                     if "ADVANCED_FACE" not in ln]
            step.write_text("\n".join(lines), encoding="latin-1")
        elif kind == "open_mesh":  # triangles dropped, header kept consistent
            data = stl.read_bytes()
            count = struct.unpack("<I", data[80:84])[0]
            keep = count - max(1, count // 20)
            stl.write_bytes(data[:80] + struct.pack("<I", keep) + data[84 : 84 + 50 * keep])


# --------------------------------------------------------------------------
# Stage 3: parse and validate
# --------------------------------------------------------------------------

def validate(rec: dict, raw_dir: Path, cfg: Config) -> tuple[dict | None, str | None]:
    """Return (parsed objects, None) for a usable part or (None, reason)."""
    try:
        ents = parse_step(raw_dir / rec["step"])
        graph = brep_graph(ents)
    except Exception as exc:  # any parser failure quarantines the part
        return None, f"step_parse_error: {type(exc).__name__}: {exc}"[:200]
    if (graph.n_faces, graph.n_edges, graph.n_vertices) != (
        rec["kernel_faces"], rec["kernel_edges"], rec["kernel_vertices"]
    ):
        return None, "topology_mismatch: parsed counts differ from CAD kernel"
    try:
        mesh = read_mesh(raw_dir / rec["stl"])
    except Exception as exc:
        return None, f"mesh_parse_error: {type(exc).__name__}: {exc}"[:200]
    report = check_mesh(mesh)
    if report.problems():
        return None, "mesh_invalid: " + "; ".join(report.problems())
    rel = abs(report.volume - rec["kernel_volume"]) / rec["kernel_volume"]
    if rel > cfg.volume_rel_tol:
        return None, f"volume_mismatch: mesh differs from B-Rep by {rel:.2%}"
    return {"graph": graph, "mesh": mesh, "report": report, "unit_scale": length_unit_scale(ents),
            "volume_rel_err": rel}, None


def geometry_signature(parsed: dict) -> str:
    """Content hash used to find duplicate parts independent of file names."""
    g, rep = parsed["graph"], parsed["report"]
    key = {
        "faces": sorted(Counter(g.face_types).items()),
        "fev": [g.n_faces, g.n_edges, g.n_vertices],
        "bbox": np.round(g.bbox, 2).tolist(),
        "volume": round(rep.volume, 1),
    }
    return hashlib.sha1(json.dumps(key).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Stage 4: split and write
# --------------------------------------------------------------------------

def assign_splits(records: list[dict], cfg: Config) -> None:
    """Stratified by family so every split sees every family in proportion."""
    rng = np.random.default_rng(cfg.seed + 2)
    by_fam: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_fam[r["family"]].append(r)
    for fam in sorted(by_fam):
        group = sorted(by_fam[fam], key=lambda r: r["part_id"])
        order = rng.permutation(len(group))
        n_train = int(round(cfg.split[0] * len(group)))
        n_val = int(round(cfg.split[1] * len(group)))
        for rank, idx in enumerate(order):
            group[idx]["split"] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"


def write_sample(rec: dict, parsed: dict, cfg: Config, out: Path, family_ids: dict) -> None:
    pts, nrm = sample_surface(parsed["mesh"], cfg.n_points, seed=cfg.seed)
    pts_n, centre, scale = normalize_unit_sphere(pts)
    g = parsed["graph"]
    np.savez_compressed(
        out / f"{rec['part_id']}.npz",
        points=pts_n,
        normals=nrm,
        centre=centre,
        scale=np.float32(scale),
        voxels=np.packbits(voxelize_points(pts_n, cfg.voxel_resolution)),
        face_features=g.node_features(),
        edge_index=g.edge_index,
        edge_features=g.edge_features(),
        family=np.int64(family_ids[rec["family"]]),
        params=np.array(list(rec["params"].values()), np.float32),
    )


def build(cfg: Config) -> dict:
    t0 = time.perf_counter()
    root = Path(cfg.out_dir)
    if root.exists():
        shutil.rmtree(root)
    raw, samples, quarantine = root / "raw", root / "samples", root / "quarantine"
    for d in (raw, samples, quarantine):
        d.mkdir(parents=True)

    jobs = plan_jobs(cfg, raw)
    # spawn, not fork: forking after the CAD kernel is loaded can deadlock
    with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=mp.get_context("spawn")) as pool:
        records = list(pool.map(_generate_one, jobs, chunksize=8))
    t_gen = time.perf_counter() - t0
    inject_faults(records, cfg, raw)

    kept, quarantined, seen = [], [], {}
    parsed_by_id = {}
    for rec in sorted(records, key=lambda r: r["part_id"]):
        parsed, reason = validate(rec, raw, cfg)
        if parsed is not None:
            sig = geometry_signature(parsed)
            if sig in seen:
                parsed, reason = None, f"duplicate_of: {seen[sig]}"
            else:
                seen[sig] = rec["part_id"]
                rec["signature"] = sig
        if parsed is None:
            rec["reason"] = reason
            quarantined.append(rec)
            for key in ("step", "stl"):
                shutil.move(raw / rec[key], quarantine / rec[key])
            continue
        rec.update(
            n_faces=parsed["graph"].n_faces,
            n_graph_edges=int(parsed["graph"].edge_index.shape[1] // 2),
            surface_types=dict(Counter(parsed["graph"].face_types)),
            mesh_triangles=parsed["report"].n_faces,
            volume_rel_err=parsed["volume_rel_err"],
        )
        parsed_by_id[rec["part_id"]] = parsed
        kept.append(rec)

    assign_splits(kept, cfg)
    family_ids = {f: i for i, f in enumerate(cfg.families)}
    for rec in kept:
        write_sample(rec, parsed_by_id[rec["part_id"]], cfg, samples, family_ids)

    drop = ("raw_dir", "stl_tolerance", "angular_tolerance")
    with open(root / "manifest.jsonl", "w") as f:
        for rec in kept:
            f.write(json.dumps({k: v for k, v in rec.items() if k not in drop}) + "\n")
    with open(root / "quarantine.jsonl", "w") as f:
        for rec in quarantined:
            f.write(json.dumps({k: v for k, v in rec.items() if k not in drop}) + "\n")

    injected = Counter(r.get("injected", "clean") for r in records)
    caught = Counter(r.get("injected", "clean") for r in quarantined)
    # A duplicate whose original was corrupted is the only valid copy left,
    # so keeping it is the correct outcome, not a miss.
    q_params = {json.dumps(r["params"], sort_keys=True) for r in quarantined if r.get("injected") != "duplicate"}
    dup_sole_copy = sum(
        1 for r in kept if r.get("injected") == "duplicate" and json.dumps(r["params"], sort_keys=True) in q_params
    )
    stats = {
        "generated": len(records),
        "kept": len(kept),
        "quarantined": len(quarantined),
        "injected": dict(injected),
        "caught": dict(caught),
        "clean_parts_wrongly_quarantined": caught.get("clean", 0),
        "duplicates_kept_as_sole_valid_copy": dup_sole_copy,
        "splits": dict(Counter(r["split"] for r in kept)),
        "families": {f: sum(r["family"] == f for r in kept) for f in cfg.families},
        "surface_types": dict(sum((Counter(r["surface_types"]) for r in kept), Counter())),
        "max_volume_rel_err": max((r["volume_rel_err"] for r in kept), default=0.0),
        "faces_per_part": [min(r["n_faces"] for r in kept), max(r["n_faces"] for r in kept)] if kept else [],
        "seconds_generation": round(t_gen, 1),
        "seconds_total": round(time.perf_counter() - t0, 1),
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()},
    }
    (root / "stats.json").write_text(json.dumps(stats, indent=2))
    (root / "report.md").write_text(render_report(stats))
    return stats


def render_report(s: dict) -> str:
    lines = [
        "# Dataset build report",
        "",
        f"- Generated parts: {s['generated']}",
        f"- Kept: {s['kept']}, quarantined: {s['quarantined']}",
        f"- Clean parts wrongly quarantined: {s['clean_parts_wrongly_quarantined']}",
        f"- Duplicates kept because their original was corrupted: {s['duplicates_kept_as_sole_valid_copy']}",
        f"- Splits: {s['splits']}",
        f"- Faces per part: {s['faces_per_part']}",
        f"- Largest mesh vs B-Rep volume error: {s['max_volume_rel_err']:.4%}",
        f"- Build time: {s['seconds_total']} s ({s['seconds_generation']} s CAD generation)",
        "",
        "## Injected faults and how many were caught",
        "",
        "| Kind | Injected | Quarantined |",
        "|---|---|---|",
    ]
    for kind, n in sorted(s["injected"].items()):
        lines.append(f"| {kind} | {n} | {s['caught'].get(kind, 0)} |")
    lines += ["", "## Surface types across kept parts", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(s["surface_types"].items())]
    return "\n".join(lines) + "\n"
