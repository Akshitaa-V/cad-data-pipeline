"""Parametric part families built with CadQuery.

Each family is a function of a parameter dict, so every generated part keeps
the exact parameters that produced it. Those parameters travel with the part
into the dataset as labels for parametric learning tasks.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np


def _cq():
    import cadquery as cq  # imported lazily: only generation needs the CAD kernel

    return cq


def plate_with_holes(p: dict):
    cq = _cq()
    w, d, t, n, hole_d = p["width"], p["depth"], p["thickness"], p["n_holes"], p["hole_d"]
    xs = np.linspace(-w / 2 + w / (n + 1), w / 2 - w / (n + 1), n) if n > 1 else [0.0]
    body = cq.Workplane("XY").box(w, d, t)
    body = body.faces(">Z").workplane().pushPoints([(float(x), 0.0) for x in xs]).hole(hole_d)
    return body.edges("|Z").fillet(p["corner_r"])


def l_bracket(p: dict):
    cq = _cq()
    a, b, t, w, n = p["leg_a"], p["leg_b"], p["thickness"], p["width"], p["n_holes"]
    profile = (
        cq.Workplane("XZ")
        .polyline([(0, 0), (a, 0), (a, t), (t, t), (t, b), (0, b)])
        .close()
        .extrude(-w)
    )
    # Mounting holes through the base leg, evenly spaced between the wall and the tip.
    xs = np.linspace(t, a, n + 2)[1:-1]
    pts = [(float(x), w / 2) for x in xs]
    holes = cq.Workplane("XY").pushPoints(pts).circle(p["hole_d"] / 2).extrude(t)
    return profile.cut(holes)


def flanged_shaft(p: dict):
    cq = _cq()
    body = cq.Workplane("XY").circle(p["flange_r"]).extrude(p["flange_t"])
    body = body.faces(">Z").workplane().circle(p["shaft_r"]).extrude(p["shaft_len"])
    body = body.faces(">Z").workplane().hole(p["bore_d"])
    return body.faces(">Z").edges().chamfer(p["chamfer"])


def stepped_block(p: dict):
    cq = _cq()
    body = cq.Workplane("XY").box(p["length"], p["width"], p["base_h"])
    body = (
        body.faces(">Z").workplane()
        .center(-p["length"] / 4, 0)
        .rect(p["length"] / 2, p["width"])
        .extrude(p["step_h"])
    )
    return body.faces(">Z").workplane().hole(p["pocket_d"], p["step_h"] * 0.8)


FAMILIES: dict[str, tuple[Callable, dict[str, tuple[float, float]], set[str]]] = {
    # name: (builder, parameter ranges, integer parameters)
    "plate_with_holes": (
        plate_with_holes,
        {"width": (30, 80), "depth": (15, 40), "thickness": (2, 8), "n_holes": (1, 4),
         "hole_d": (2, 5), "corner_r": (0.5, 3)},
        {"n_holes"},
    ),
    "l_bracket": (
        l_bracket,
        {"leg_a": (30, 60), "leg_b": (20, 50), "thickness": (3, 8), "width": (12, 30),
         "hole_d": (3, 5), "n_holes": (1, 3)},
        {"n_holes"},
    ),
    "flanged_shaft": (
        flanged_shaft,
        {"flange_r": (12, 25), "flange_t": (3, 8), "shaft_r": (4, 10), "shaft_len": (15, 50),
         "bore_d": (1.5, 3.5), "chamfer": (0.3, 1.0)},
        set(),
    ),
    "stepped_block": (
        stepped_block,
        {"length": (30, 70), "width": (15, 35), "base_h": (5, 15), "step_h": (5, 15),
         "pocket_d": (4, 10)},
        set(),
    ),
}


def sample_params(family: str, rng: np.random.Generator) -> dict:
    _, ranges, ints = FAMILIES[family]
    out = {}
    for k, (lo, hi) in ranges.items():
        out[k] = int(rng.integers(lo, hi + 1)) if k in ints else round(float(rng.uniform(lo, hi)), 2)
    return out


def build_part(family: str, params: dict):
    return FAMILIES[family][0](params)


def export_part(shape, stem: Path, stl_tolerance: float, angular_tolerance: float) -> dict:
    """Write STEP and STL next to each other; return kernel reference values
    that the pipeline later uses to cross-check its own parsers."""
    cq = _cq()
    step_path, stl_path = stem.with_suffix(".step"), stem.with_suffix(".stl")
    cq.exporters.export(shape, str(step_path))
    cq.exporters.export(shape, str(stl_path), tolerance=stl_tolerance, angularTolerance=angular_tolerance)
    solid = shape.val()
    return {
        "step": step_path.name,
        "stl": stl_path.name,
        "kernel_faces": len(shape.faces().vals()),
        "kernel_edges": len(shape.edges().vals()),
        "kernel_vertices": len(shape.vertices().vals()),
        "kernel_volume": float(solid.Volume()),
        "kernel_area": float(solid.Area()),
    }
