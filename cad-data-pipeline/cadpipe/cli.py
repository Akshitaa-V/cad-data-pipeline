"""Command line entry point.

    python -m cadpipe build --config configs/default.yaml
    python -m cadpipe inspect path/to/part.step
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from .dataset import Config, build
from .mesh_io import check_mesh, read_mesh
from .step_parser import brep_graph, parse_step


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cadpipe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the dataset from a YAML config")
    b.add_argument("--config", default="configs/default.yaml")
    i = sub.add_parser("inspect", help="print topology or mesh checks for one file")
    i.add_argument("path")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        stats = build(Config.load(args.config))
        print(json.dumps({k: v for k, v in stats.items() if k != "config"}, indent=2))
        return 0

    if args.path.lower().endswith((".step", ".stp")):
        g = brep_graph(parse_step(args.path))
        print(json.dumps({
            "faces": g.n_faces, "edges": g.n_edges, "vertices": g.n_vertices,
            "adjacency_pairs": int(g.edge_index.shape[1] // 2),
            "surface_types": dict(Counter(g.face_types)),
            "bbox_mm": g.bbox.round(3).tolist(),
        }, indent=2))
    else:
        rep = check_mesh(read_mesh(args.path))
        print(json.dumps({**rep.__dict__, "problems": rep.problems()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
