from pathlib import Path

import numpy as np

from tinycwrap import CModule


def test_dot_and_scale_example():
    c_path = Path(__file__).resolve().parent.parent / "examples" / "kernels.c"
    cm = CModule(c_path)

    x = np.arange(10, dtype=np.float64)
    y = np.ones_like(x)

    # dot example
    assert cm.dot(x, y) == np.sum(x * y)

    # scale example (auto output for out_x)
    scaled = cm.scale(x, 1.1)
    np.testing.assert_allclose(scaled, x * 1.1)
