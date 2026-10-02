# cad-data-pipeline

A Python pipeline that turns CAD files into clean, validated training data for 3D deep learning.

It generates parametric parts, parses STEP and STL/OBJ files with its own parsers, checks every part for broken geometry, removes duplicates, and writes each part as a point cloud with normals, a voxel grid and a B-Rep face adjacency graph, together with the parameters that produced it.

```
CadQuery part families ──► STEP + STL ──► parse ──► validate ──► deduplicate ──► split ──► .npz shards
       (parameters)                         │           │                                   + manifest.jsonl
                                            └──── quarantine/ with a reason ◄──────┘        + report.md
```

## Results of the default build

`python -m cadpipe build --config configs/default.yaml` on a laptop CPU, 4 workers:

| | |
|---|---|
| Parts generated | 420 (4 families × 100, plus 20 duplicates) |
| Parts kept | 381 (train 305 / val 39 / test 37, stratified by family) |
| Injected faults caught | 20 of 20 (7 truncated STL, 7 broken STEP, 6 open meshes) |
| Duplicates removed | 19 of 20, see below |
| Clean parts wrongly quarantined | 0 of 380 |
| STEP parser vs CAD kernel | face, edge and vertex counts identical on all 380 clean parts |
| Mesh volume vs B-Rep volume | largest difference 0.042 % |
| Build time | 23.6 s, of which 15.8 s is CAD generation |

The duplicate that was kept is not a miss: its original had a corrupted STEP file and was quarantined, so the duplicate was the only valid copy left. The build report counts this case separately.

## What it does

**STEP parser (`cadpipe/step_parser.py`).** Reads ISO 10303-21 files (AP203/AP214) without a CAD kernel: quoted strings with `;` and escaped quotes, comments, multi-line records, typed parameters and complex entities such as `( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )`. It then walks the B-Rep from `ADVANCED_FACE` through `FACE_BOUND`, `EDGE_LOOP`, `ORIENTED_EDGE` and `EDGE_CURVE` down to `VERTEX_POINT`, and builds a face adjacency graph: one node per face, one edge between two faces that share an edge curve. Seam edges, which join a cylindrical face to itself, are not treated as adjacency.

Node features are the surface type (plane, cylinder, cone, sphere, torus, B-spline, other), the number of boundary loops (1 plus one per hole) and the number of edges. Edge features are the curve type (line, circle, ellipse, B-spline, other). This is the input format of graph models for B-Reps such as UV-Net and BRepNet, without the UV-grid samples.

**Mesh readers (`cadpipe/mesh_io.py`).** Binary and ASCII STL and OBJ (polygons are fan-triangulated). Duplicate corners are welded so that shared edges become shared indices, which the checks rely on. A binary STL whose size does not match its triangle count is rejected instead of being read as garbage.

**Mesh checks.** Degenerate triangles, watertightness (every edge used by exactly two triangles), signed volume (negative means inverted normals), surface area and Euler characteristic.

**Representations (`cadpipe/sampling.py`).** Area-weighted surface sampling with outward normals, normalisation into the unit sphere that stores centre and scale so original coordinates can be recovered, and voxel occupancy grids.

**Part generator (`cadpipe/generate.py`).** Four parametric families built in CadQuery: plates with 1 to 4 holes, L-brackets with 1 to 3 mounting holes, flanged shafts with a bore and chamfer, and stepped blocks with a pocket. Each part is exported as STEP and STL, and the kernel's own face, edge and vertex counts and volume are stored as reference values.

**Dataset build (`cadpipe/dataset.py`).** One YAML file controls everything. The build injects known faults on purpose, so the validators are tested against ground truth rather than assumed to work:

| Fault | Simulates | Caught by |
|---|---|---|
| `truncated_stl` | interrupted download | STL size check |
| `broken_step` | exporter dropping faces | STEP parser |
| `open_mesh` | triangles missing, header still consistent | watertightness check |
| `duplicate` | same part uploaded twice under a new name | geometry signature |

A part is kept only if its STEP topology matches the kernel exactly, its mesh is watertight with positive volume, and the mesh volume is within 1 % of the B-Rep volume. Everything else moves to `quarantine/` with the reason in `quarantine.jsonl`, so no file disappears without a trace.

## Output

```
data/
  samples/<part_id>.npz   points, normals, centre, scale, voxels (bit-packed),
                          face_features, edge_index, edge_features, family, params
  manifest.jsonl          one row per kept part: family, parameters, split, signature, counts
  quarantine.jsonl        one row per rejected part, with the reason
  quarantine/             the rejected STEP and STL files
  raw/                    the accepted STEP and STL files
  stats.json, report.md
```

`cadpipe.loader.CadDataset(root, split="train")` reads it back as dictionaries of numpy arrays, so it works with PyTorch, JAX or plain numpy without extra dependencies.

## Usage

```bash
pip install -e ".[cad,dev]"

python -m cadpipe build --config configs/default.yaml   # full dataset
python -m cadpipe build --config configs/smoke.yaml     # small build for CI
python -m cadpipe inspect tests/fixtures/plate_hole.step
pytest -q
```

Parsing, checks and sampling need only numpy and PyYAML. CadQuery is needed only to generate parts.

## Tests

40 tests, run in GitHub Actions on Python 3.11 and 3.12:

- parser edge cases: strings with semicolons and escaped quotes, comments, multi-line records, typed parameters, complex entities, invalid files
- topology of known shapes: a box has 6 faces, 12 edges, 8 vertices and 4 neighbours per face; a cylinder's seam edge is not adjacency; a hole adds a second loop to two faces
- parser against the CAD kernel on parts from every family
- mesh readers, watertightness, degenerate and inverted triangles
- sampling lies on the surface, is area-weighted (checked on a 1:3 area split) and is reproducible; normalisation is reversible
- end to end: every injected fault is quarantined, no clean part is, no duplicate geometry survives, splits are disjoint and stratified, and a rebuild with the same config gives an identical manifest

## Limits

- The parts are synthetic and simple: 8 to 14 faces with planar, cylindrical and conical surfaces. Real datasets such as ABC or Fusion 360 Gallery contain free-form B-spline surfaces and much larger models. The parser reads B-spline entities, but this dataset does not exercise them.
- The STEP parser recovers topology and surface or curve types. It does not evaluate surface geometry, so it cannot produce UV-grid samples. That needs a CAD kernel.
- One solid per file. Assemblies and multi-body files are not handled.
- The volume check compares against kernel values recorded at generation time. For files that come from outside the pipeline, only the topology and mesh checks apply.

## License

MIT
