from pathlib import Path

import numpy as np

import pytest

from tinycwrap import CModule


@pytest.fixture(scope="module")
def cm():
    c_path = Path(__file__).resolve().parent / "test_kernels.c"
    return CModule(c_path)


def test_dot(cm):
    x = np.arange(10, dtype=np.float64)
    y = np.ones_like(x)
    assert cm.dot(x, y, len_x=len(x)) == np.sum(x * y)


def test_scale_auto_output(cm):
    x = np.arange(10, dtype=np.float64)
    scaled_auto = cm.scale(x, 1.1, len_x=len(x))
    np.testing.assert_allclose(scaled_auto, x * 1.1)


def test_scale_explicit_output(cm):
    x = np.arange(10, dtype=np.float64)
    out = np.empty_like(x)
    scaled_explicit = cm.scale(x, 2.0, len_x=len(x), out_x=out)
    np.testing.assert_allclose(out, x * 2.0)
    np.testing.assert_allclose(scaled_explicit, x * 2.0)


def test_struct_wrapper(cm):
    cp = cm.ComplexPair(real=1.5, imag=-2.5)
    assert repr(cp) == "ComplexPair(real=1.5, imag=-2.5)"
    assert cp.real == 1.5
    assert cp.imag == -2.5
    cp.real = 3.0
    cp.imag = 4.0
    assert cp.real == 3.0
    assert cp.imag == 4.0
    np.testing.assert_equal(cp.dtype.fields.keys(), {"real", "imag"})
    assert cp.dtype["real"] == np.dtype(np.float64)


def test_struct_argument(cm):
    cp = cm.ComplexPair(real=3.0, imag=4.0)
    mag_sq = cm.complex_magnitude(cp)
    assert mag_sq == 5.0
