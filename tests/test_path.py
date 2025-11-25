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
