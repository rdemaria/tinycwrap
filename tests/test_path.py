from pathlib import Path

import numpy as np
import pytest

from tinycwrap import CModule


@pytest.fixture(scope="module")
def cp():
    return CModule(Path("tests/t1/base.c"), Path("tests/t1/path.c"))


def test_path_get_steps_contract(cp):
    seg = cp.G2DSegment()
    # line from (0,0) to (1,0)
    seg.type = 0
    seg.data = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    segments = np.array([seg._data], dtype=cp.G2DSegment.dtype)
    steps = cp.geom2d_path_get_steps(segments, ds_min=0.25)
    expected_len = cp.geom2d_path_get_len_steps(segments, len_segments=len(segments), ds_min=0.25)
    assert len(steps) == expected_len
    assert np.isclose(steps[-1], cp.geom2d_path_get_length(segments, len_segments=1))


def test_path_struct_pointer_array(cp):
    path = cp.G2DPath(len_segments=3)
    assert path.len_segments == 3
    assert isinstance(path.segments, np.ndarray)
    assert path.segments.shape == (3,)
    assert path.segments.dtype == cp.G2DSegment.dtype


def test_return_class_path(cp):
    path=cp.G2DPath(len_segments=1)
    cp.geom2d_path_from_circle(1.0, path)
    assert path.len_segments == 1
    assert path.segments.shape == (1,)
    seg = path.segments[0]
    assert seg['type'] == 1  # circle segment
    np.testing.assert_allclose(seg['data'][:3], [0.0, 0.0, 1.0])  # cx,cy,radius

def test_build_pointer_from_data(cp):
    segments=cp.geom2d_rectangle_to_path(2.0, 3.0)
    assert segments.shape == (4,)
    path=cp.G2DPath(segments=segments, len_segments=len(segments))
    assert path.len_segments == 4
    assert path.segments.shape == (4,)

def test_repr_pointer_array(cp):
    path = cp.G2DPath(len_segments=2)
    repr_str = repr(path)
    assert "segments=<segments* 0x" in repr_str
    assert "len_segments=2" in repr_str
