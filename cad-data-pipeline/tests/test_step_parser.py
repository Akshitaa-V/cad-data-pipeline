from pathlib import Path

import numpy as np
import pytest

from cadpipe.step_parser import (
    Enum,
    Ref,
    StepParseError,
    brep_graph,
    length_unit_scale,
    parse_step,
    parse_step_text,
)

FIX = Path(__file__).parent / "fixtures"


def _wrap(data: str) -> str:
    return "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n" + data + "\nENDSEC;\nEND-ISO-10303-21;\n"


def test_reads_references_numbers_enums_and_unset_values():
    ents = parse_step_text(_wrap("#1 = THING('a', #2, 1.5E-3, -4, .T., $, *, (1., 2.));"))
    assert ents[1].type == "THING"
    assert ents[1].args == ["a", Ref(2), 1.5e-3, -4, Enum("T"), None, None, [1.0, 2.0]]


def test_semicolon_and_escaped_quote_inside_string_do_not_split():
    ents = parse_step_text(_wrap("#1 = NOTE('it''s; fine');\n#2 = NOTE('x');"))
    assert ents[1].args == ["it's; fine"]
    assert set(ents) == {1, 2}


def test_statement_spanning_lines_and_comments():
    ents = parse_step_text(_wrap("/* comment; with semicolon */\n#7 = PT('',\n (0.,\n 1., 2.));"))
    assert ents[7].args[1] == [0.0, 1.0, 2.0]


def test_typed_parameter():
    ents = parse_step_text(_wrap("#3 = MEASURE(LENGTH_MEASURE(2.5),#4);"))
    assert ents[3].args[0] == ("LENGTH_MEASURE", [2.5])


def test_complex_entity_keeps_all_parts():
    ents = parse_step_text(_wrap("#9 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );"))
    assert set(ents[9].parts) == {"LENGTH_UNIT", "NAMED_UNIT", "SI_UNIT"}
    assert ents[9].parts["SI_UNIT"] == [Enum("MILLI"), Enum("METRE")]


@pytest.mark.parametrize("text", ["not a step file", "ISO-10303-21;\nHEADER;\nENDSEC;\n", _wrap("")])
def test_rejects_invalid_files(text):
    with pytest.raises(StepParseError):
        parse_step_text(text)


def test_box_topology():
    g = brep_graph(parse_step(FIX / "box.step"))
    assert (g.n_faces, g.n_edges, g.n_vertices) == (6, 12, 8)
    assert g.edge_index.shape == (2, 24)            # 12 adjacent face pairs, both directions
    assert np.bincount(g.edge_index[0]).tolist() == [4] * 6  # every box face touches four others
    assert set(g.face_types) == {"PLANE"}
    assert g.euler_characteristic() == 2
    np.testing.assert_allclose(g.bbox, [[-5, -5, -5], [5, 5, 5]], atol=1e-9)


def test_cylinder_seam_is_not_adjacency():
    g = brep_graph(parse_step(FIX / "cylinder.step"))
    assert sorted(g.face_types) == ["CYLINDRICAL_SURFACE", "PLANE", "PLANE"]
    # side touches top and bottom; the seam edge joins the side to itself only
    assert g.edge_index.shape == (2, 4)
    assert not np.any(g.edge_index[0] == g.edge_index[1])


def test_hole_gives_a_second_loop():
    g = brep_graph(parse_step(FIX / "plate_hole.step"))
    assert g.n_faces == 7
    assert sorted(g.face_loops)[-2:] == [2, 2]  # top and bottom faces carry the hole
    assert g.face_types.count("CYLINDRICAL_SURFACE") == 1


def test_feature_shapes():
    g = brep_graph(parse_step(FIX / "plate_hole.step"))
    nf, ef = g.node_features(), g.edge_features()
    assert nf.shape == (7, 9) and nf.dtype == np.float32
    assert np.all(nf[:, :7].sum(1) == 1)
    assert ef.shape[0] == g.edge_index.shape[1]


def test_unit_scale_to_millimetres():
    assert length_unit_scale(parse_step(FIX / "box.step")) == 1.0
    metre = parse_step_text(_wrap("#1 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.) );"))
    assert length_unit_scale(metre) == 1000.0


def test_file_without_faces_is_rejected():
    ents = parse_step_text(_wrap("#1 = CARTESIAN_POINT('',(0.,0.,0.));"))
    with pytest.raises(StepParseError, match="no ADVANCED_FACE"):
        brep_graph(ents)
