from pathlib import Path

import numpy as np

from tinycwrap import CModule


def test_dot_and_scale_example():
    c_path = Path(__file__).resolve().parent / "kernels.c"
    cm = CModule(c_path)

    x = np.arange(10, dtype=np.float64)
    y = np.ones_like(x)

    # dot example
    assert cm.dot(x, y) == np.sum(x * y)

    # scale example (auto output for out_x)
    scaled_auto = cm.scale(x, 1.1)
    np.testing.assert_allclose(scaled_auto, x * 1.1)

    # scale example with explicit output array
    out = np.empty_like(x)
    scaled_explicit = cm.scale(x, 2.0, out_x=out)
    np.testing.assert_allclose(out, x * 2.0)
    np.testing.assert_allclose(scaled_explicit, x * 2.0)
