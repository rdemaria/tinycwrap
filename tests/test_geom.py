from pathlib import Path

from tinycwrap import CModule


def test_geom2d():
    cg = CModule(Path(__file__).resolve().parent / "cgeom.c")
    pts = cg.geom2d_circle_get_points(0.0, 0.0, 1.0)
    assert pts.shape == (101,)
    # first point at angle 0 -> (1,0)
    assert pts["x"][0] == 1.0
    assert abs(pts["y"][0]) < 1e-12
