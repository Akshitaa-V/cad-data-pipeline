"""Read a built dataset back. Framework-free, so it plugs into PyTorch,
JAX or plain numpy code without extra dependencies."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class CadDataset:
    def __init__(self, root: str | Path, split: str | None = None):
        self.root = Path(root)
        rows = [json.loads(ln) for ln in (self.root / "manifest.jsonl").read_text().splitlines() if ln]
        self.rows = [r for r in rows if split is None or r["split"] == split]
        stats = json.loads((self.root / "stats.json").read_text())
        self.voxel_resolution = stats["config"]["voxel_resolution"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        with np.load(self.root / "samples" / f"{row['part_id']}.npz") as z:
            item = {k: z[k] for k in z.files}
        r = self.voxel_resolution
        item["voxels"] = np.unpackbits(item["voxels"])[: r ** 3].reshape(r, r, r)
        item["part_id"] = row["part_id"]
        return item
