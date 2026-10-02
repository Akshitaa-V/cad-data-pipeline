"""Parser for STEP files (ISO 10303-21, AP203/AP214) that recovers B-Rep topology.

The parser reads the DATA section into a table of entities and then walks the
boundary representation: ADVANCED_FACE -> FACE_BOUND -> EDGE_LOOP ->
ORIENTED_EDGE -> EDGE_CURVE -> VERTEX_POINT -> CARTESIAN_POINT.

From that walk it builds a face adjacency graph: one node per face, one edge
between two faces that share an EDGE_CURVE. This is the graph representation
used by B-Rep learning models such as UV-Net and BRepNet.

No CAD kernel is needed, so the parser runs anywhere numpy runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class StepParseError(ValueError):
    """Raised when a file is not a readable ISO 10303-21 exchange file."""


@dataclass(frozen=True)
class Ref:
    """A reference to another entity, written #123 in the file."""

    id: int


@dataclass(frozen=True)
class Enum:
    """An enumeration value, written .T. or .UNSPECIFIED. in the file."""

    value: str


@dataclass
class Entity:
    id: int
    type: str
    args: list
    # Complex entities, written #5 = ( A() B() C() ); keep every part here.
    parts: dict[str, list] = field(default_factory=dict)


# Surface and curve types the graph encodes as one-hot features.
SURFACE_TYPES = [
    "PLANE",
    "CYLINDRICAL_SURFACE",
    "CONICAL_SURFACE",
    "SPHERICAL_SURFACE",
    "TOROIDAL_SURFACE",
    "B_SPLINE_SURFACE",
    "OTHER_SURFACE",
]
CURVE_TYPES = ["LINE", "CIRCLE", "ELLIPSE", "B_SPLINE_CURVE", "OTHER_CURVE"]


# --------------------------------------------------------------------------
# Tokenising and argument parsing
# --------------------------------------------------------------------------

def _split_statements(data: str) -> list[str]:
    """Split the DATA section on ';' while respecting quoted strings."""
    out, buf, in_str, i = [], [], False, 0
    while i < len(data):
        c = data[i]
        if in_str:
            buf.append(c)
            if c == "'":
                if i + 1 < len(data) and data[i + 1] == "'":  # escaped quote
                    buf.append("'")
                    i += 1
                else:
                    in_str = False
        elif c == "'":
            in_str = True
            buf.append(c)
        elif c == "/" and data.startswith("/*", i):  # comment
            end = data.find("*/", i + 2)
            i = len(data) if end < 0 else end + 2
            continue
        elif c == ";":
            stmt = "".join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
        else:
            buf.append(c)
        i += 1
    return out


class _ArgReader:
    """Recursive-descent reader for one parenthesised argument list."""

    def __init__(self, text: str):
        self.s = text
        self.i = 0

    def _ws(self) -> None:
        while self.i < len(self.s) and self.s[self.i] in " \t\r\n":
            self.i += 1

    def _peek(self) -> str:
        self._ws()
        return self.s[self.i] if self.i < len(self.s) else ""

    def _expect(self, ch: str) -> None:
        if self._peek() != ch:
            raise StepParseError(f"expected '{ch}' at {self.i} in {self.s[:80]!r}")
        self.i += 1

    def read_list(self) -> list:
        self._expect("(")
        items: list = []
        if self._peek() == ")":
            self.i += 1
            return items
        while True:
            items.append(self.read_value())
            c = self._peek()
            self.i += 1
            if c == ")":
                return items
            if c != ",":
                raise StepParseError(f"expected ',' or ')' at {self.i} in {self.s[:80]!r}")

    def read_value(self):
        c = self._peek()
        s = self.s
        if c == "(":
            return self.read_list()
        if c == "#":
            j = self.i + 1
            while j < len(s) and s[j].isdigit():
                j += 1
            val = Ref(int(s[self.i + 1 : j]))
            self.i = j
            return val
        if c == "'":
            j, buf = self.i + 1, []
            while j < len(s):
                if s[j] == "'":
                    if j + 1 < len(s) and s[j + 1] == "'":
                        buf.append("'")
                        j += 2
                        continue
                    break
                buf.append(s[j])
                j += 1
            self.i = j + 1
            return "".join(buf)
        if c == ".":
            j = s.index(".", self.i + 1)
            val = Enum(s[self.i + 1 : j])
            self.i = j + 1
            return val
        if c in "$*":
            self.i += 1
            return None
        if c == '"':  # binary literal, not used for geometry
            j = s.index('"', self.i + 1)
            self.i = j + 1
            return s[self.i - 1 : j]
        if c.isalpha():  # typed parameter, e.g. LENGTH_MEASURE(1.)
            j = self.i
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            name = s[self.i : j]
            self.i = j
            inner = self.read_list()
            return (name, inner)
        # number
        j = self.i
        while j < len(s) and s[j] in "+-0123456789.Ee":
            j += 1
        tok = s[self.i : j]
        if not tok:
            raise StepParseError(f"unexpected character {c!r} at {self.i}")
        self.i = j
        try:
            return int(tok) if tok.lstrip("+-").isdigit() else float(tok)
        except ValueError as exc:
            raise StepParseError(f"bad number {tok!r}") from exc


def _parse_record(body: str) -> tuple[str, list, dict[str, list]]:
    body = body.strip()
    if body.startswith("("):  # complex entity
        r = _ArgReader(body)
        r._expect("(")
        parts: dict[str, list] = {}
        while r._peek() not in (")", ""):
            j = r.i
            while j < len(r.s) and (r.s[j].isalnum() or r.s[j] == "_"):
                j += 1
            name = r.s[r.i : j]
            if not name:
                raise StepParseError(f"bad complex entity {body[:80]!r}")
            r.i = j
            parts[name] = r.read_list()
        primary = next(iter(parts)) if parts else "UNKNOWN"
        # Use the most specific geometric part as the type when present.
        for name in parts:
            if name.startswith(("B_SPLINE_SURFACE", "B_SPLINE_CURVE", "BOUNDED_")):
                primary = name
                break
        return primary, parts.get(primary, []), parts
    k = body.index("(")
    name = body[:k].strip()
    args = _ArgReader(body[k:]).read_list()
    return name, args, {name: args}


def parse_step_text(text: str) -> dict[int, Entity]:
    if "ISO-10303-21" not in text[:200]:
        raise StepParseError("missing ISO-10303-21 header")
    start = text.find("DATA;")
    end = text.rfind("ENDSEC;")
    if start < 0 or end < start:
        raise StepParseError("missing DATA section")
    entities: dict[int, Entity] = {}
    for stmt in _split_statements(text[start + 5 : end]):
        if not stmt.startswith("#"):
            continue
        eq = stmt.index("=")
        eid = int(stmt[1:eq].strip())
        etype, args, parts = _parse_record(stmt[eq + 1 :])
        entities[eid] = Entity(eid, etype, args, parts)
    if not entities:
        raise StepParseError("DATA section has no entities")
    return entities


def parse_step(path: str | Path) -> dict[int, Entity]:
    text = Path(path).read_text(encoding="latin-1")
    return parse_step_text(text)


# --------------------------------------------------------------------------
# B-Rep topology
# --------------------------------------------------------------------------

def _surface_kind(t: str) -> str:
    if t.startswith(("B_SPLINE_SURFACE", "BOUNDED_SURFACE")):
        return "B_SPLINE_SURFACE"
    return t if t in SURFACE_TYPES else "OTHER_SURFACE"


def _curve_kind(t: str) -> str:
    if t.startswith(("B_SPLINE_CURVE", "BOUNDED_CURVE")):
        return "B_SPLINE_CURVE"
    return t if t in CURVE_TYPES else "OTHER_CURVE"


@dataclass
class BRepGraph:
    """Face adjacency graph of one solid."""

    face_ids: list[int]
    face_types: list[str]
    face_loops: list[int]          # number of boundary loops per face (1 + holes)
    face_edges: list[int]          # number of oriented edges per face
    edge_index: np.ndarray         # (2, E) undirected, both directions stored
    edge_types: list[str]          # curve type per undirected edge (len E/2)
    n_vertices: int
    n_edges: int                   # distinct EDGE_CURVE entities
    bbox: np.ndarray               # (2, 3) from vertex coordinates

    @property
    def n_faces(self) -> int:
        return len(self.face_ids)

    def node_features(self) -> np.ndarray:
        """One-hot surface type + loop count + edge count, float32 (F, 9)."""
        onehot = np.zeros((self.n_faces, len(SURFACE_TYPES)), np.float32)
        for i, t in enumerate(self.face_types):
            onehot[i, SURFACE_TYPES.index(t)] = 1.0
        extra = np.array([self.face_loops, self.face_edges], np.float32).T
        return np.concatenate([onehot, extra], axis=1)

    def edge_features(self) -> np.ndarray:
        """One-hot curve type per directed edge, float32 (E, 5)."""
        feats = np.zeros((len(self.edge_types), len(CURVE_TYPES)), np.float32)
        for i, t in enumerate(self.edge_types):
            feats[i, CURVE_TYPES.index(t)] = 1.0
        return np.repeat(feats, 2, axis=0)  # matches both stored directions

    def euler_characteristic(self) -> int:
        """V - E + F; equals 2 - 2*genus for a closed orientable solid when
        every face is simply connected. Inner loops lower it by one each."""
        return self.n_vertices - self.n_edges + self.n_faces


def _resolve_curve(entities: dict[int, Entity], ref: Ref) -> str:
    """EDGE_CURVE geometry may be wrapped in SURFACE_CURVE / SEAM_CURVE."""
    ent = entities[ref.id]
    seen = 0
    while ent.type in ("SURFACE_CURVE", "SEAM_CURVE", "INTERSECTION_CURVE") and seen < 5:
        ent = entities[ent.args[1].id]
        seen += 1
    return _curve_kind(ent.type)


def brep_graph(entities: dict[int, Entity]) -> BRepGraph:
    faces = sorted(e.id for e in entities.values() if e.type in ("ADVANCED_FACE", "FACE_SURFACE"))
    if not faces:
        raise StepParseError("no ADVANCED_FACE entities: file holds no B-Rep solid")

    face_types, face_loops, face_edges = [], [], []
    edge_to_faces: dict[int, set[int]] = {}
    vertex_ids: set[int] = set()

    for fi, fid in enumerate(faces):
        face = entities[fid]
        bounds, surf = face.args[1], face.args[2]
        face_types.append(_surface_kind(entities[surf.id].type))
        face_loops.append(len(bounds))
        n_oe = 0
        for b in bounds:
            loop = entities[entities[b.id].args[1].id]
            if loop.type == "VERTEX_LOOP":
                vertex_ids.add(loop.args[1].id)
                continue
            for oe_ref in loop.args[1]:
                oe = entities[oe_ref.id]
                ec_id = oe.args[3].id
                ec = entities[ec_id]
                edge_to_faces.setdefault(ec_id, set()).add(fi)
                vertex_ids.add(ec.args[1].id)
                vertex_ids.add(ec.args[2].id)
                n_oe += 1
        face_edges.append(n_oe)

    pairs: dict[tuple[int, int], str] = {}
    for ec_id, fset in edge_to_faces.items():
        fl = sorted(fset)
        ctype = _resolve_curve(entities, entities[ec_id].args[3])
        for a in range(len(fl)):
            for b in range(a + 1, len(fl)):
                pairs.setdefault((fl[a], fl[b]), ctype)
        # an edge used by a single face is a seam (e.g. a cylinder's), not adjacency

    src, dst, etypes = [], [], []
    for (a, b), ctype in sorted(pairs.items()):
        src += [a, b]
        dst += [b, a]
        etypes.append(ctype)
    edge_index = np.array([src, dst], dtype=np.int64).reshape(2, -1)

    coords = []
    for vid in vertex_ids:
        pt = entities[entities[vid].args[1].id]
        coords.append(pt.args[1])
    xyz = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    bbox = np.stack([xyz.min(0), xyz.max(0)]) if len(xyz) else np.zeros((2, 3))

    return BRepGraph(
        face_ids=faces,
        face_types=face_types,
        face_loops=face_loops,
        face_edges=face_edges,
        edge_index=edge_index,
        edge_types=etypes,
        n_vertices=len(vertex_ids),
        n_edges=len(edge_to_faces),
        bbox=bbox,
    )


def length_unit_scale(entities: dict[int, Entity]) -> float:
    """Scale factor to millimetres from the file's SI_UNIT for length."""
    prefixes = {"MILLI": 1.0, "CENTI": 10.0, "DECI": 100.0, None: 1000.0, "MICRO": 1e-3}
    for e in entities.values():
        if "LENGTH_UNIT" in e.parts and "SI_UNIT" in e.parts:
            prefix = e.parts["SI_UNIT"][0]
            return prefixes.get(prefix.value if isinstance(prefix, Enum) else None, 1.0)
    return 1.0
