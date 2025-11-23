from pathlib import Path

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


