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
    assert np.isclose(mag_sq, 5.0)


def test_struct_array_member(cm):
    p = cm.Particle()
    p.pos = [1.0, 2.0, 3.0]
    p.vel = [0.5, 0.5, 0.5]
    np.testing.assert_allclose(p.pos, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(p.vel, np.array([0.5, 0.5, 0.5]))
    p.pos[:] *= 2.0
    np.testing.assert_allclose(p.pos, np.array([2.0, 4.0, 6.0]))


def test_struct_array_argument(cm):
    particles = cm.Particle.zeros(2)
    particles["pos"][0] = [0.0, 0.0, 0.0]
    particles["vel"][0] = [1.0, 0.0, 0.0]
    particles["pos"][1] = [0.0, 0.0, 0.0]
    particles["vel"][1] = [0.0, 2.0, 0.0]
    ke = cm.kinetic_energy(particles, len_p=len(particles))
    assert np.isclose(ke, 0.5 * (1.0 + 4.0))


def test_two_output_arrays(cm):
    data = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
    out1, out2 = cm.split_vectors(data)
    np.testing.assert_allclose(out1, np.array([0, 2, 4], dtype=np.float64))
    np.testing.assert_allclose(out2, np.array([1, 3, 5], dtype=np.float64))


def test_merge_sorted(cm):
    a = np.array([1.0, 2.0, 2.0, 4.0], dtype=np.float64)
    b = np.array([2.0, 3.0], dtype=np.float64)
    out_len = np.zeros(1, dtype=np.int32)
    merged, out_len_val = cm.merge_sorted(a, b, out_len=out_len)
    np.testing.assert_allclose(merged, np.array([1.0, 2.0, 3.0, 4.0]))
    assert out_len_val == 4


def test_struct_output_array(cm):
    n = 4
    particles = cm.Particle.zeros(n)
    out_particles = cm.make_particles(3.0, out_p=particles, len_p=n)
    np.testing.assert_array_equal(out_particles, particles)
    np.testing.assert_allclose(particles["pos"], np.ones((n, 3)))
    np.testing.assert_allclose(particles["vel"], np.array([[3.0, 0.0, 0.0]] * n))
