from pathlib import Path

import numpy as np

from tinycwrap import CModule
import pytest

@pytest.fixture(scope="module")
def cg():
    c_path = Path(__file__).resolve().parent / "cgeom.c"
    return CModule(c_path)


def test_geom2d(cg):
    pts = cg.geom2d_circle_get_points(0.0, 0.0, 1.0)
    assert pts.shape == (101,)
    # first point at angle 0 -> (1,0)
    assert pts["x"][0] == 1.0
    assert abs(pts["y"][0]) < 1e-12

def test_geom2d_norm(cg):
    n = cg.geom2d_norm(3.0, 4.0)
    assert n == 5.0

def test_circle_points_length_contract(cg):
    pts = cg.geom2d_circle_get_points(1.0, 2.0, 3.0)
    assert pts.shape == (101,)


def test_return_struct_array_member(cg):
    seg = cg.G2DSegment()
    seg.type = 1
    seg.data = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    assert seg.type == 1
    np.testing.assert_allclose(seg.data, [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])

def test_geom2d_return_strct(cg):
    seg = cg.geom2d_line_segment_from_start_length(0.1, 0.2, 0.5, 0.4, 0.3)
    assert seg.type == 0
    assert seg.data[0] == 0.1
    assert seg.data[1] == 0.2
    assert np.isclose(seg.data[2], 0.1+0.3*0.5)
    assert np.isclose(seg.data[3], 0.2+0.4*0.3)


def test_geom2d_return_strct_array_member(cg):
    seg = cg.geom2d_rectangle_to_path(1,2)
    assert len(seg) == 4


def test_docstring_contains_contract(cg):
    doc = cg.geom2d_rectangle_to_path.__doc__
    assert "Contract: len(out_segments)=4" in doc
    assert "Auto-wrapped C function `geom2d_rectangle_to_path`." in doc
