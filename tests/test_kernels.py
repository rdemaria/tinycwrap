from pathlib import Path

import numpy as np

import pytest

from tinycwrap import CModule


@pytest.fixture(scope="module")
def cm():
    c_path = Path(__file__).resolve().parent / "test_kernels.c"
    return CModule(c_path)


def test_dot(cm):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    assert cm.dot(x, y, len_x=len(x)) == np.dot(x, y)


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


def test_mat_add_shape_contract(cm):
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.ones_like(a)
    out = cm.mat_add(a, b)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, a + b)

    flat = np.zeros(a.size, dtype=np.float64)
    reshaped = cm.mat_add(a, b, out=flat)
    assert reshaped.shape == (2, 3)
    np.testing.assert_allclose(reshaped, a + b)
    np.testing.assert_allclose(flat.reshape(a.shape), a + b)


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


def test_struct_pointer_inplace_no_return(cm):
    cp = cm.ComplexPair(real=2.0, imag=-3.0)
    res = cm.scale_complex(cp, 2.0)
    assert res is None
    assert cp.real == 4.0
    assert cp.imag == -6.0


def test_struct_scalar_output_returns_struct_object(cm):
    a = cm.ComplexPair(real=2.0, imag=4.0)
    b = cm.ComplexPair(real=6.0, imag=10.0)
    mid = cm.midpoint_complex(a, b)
    assert isinstance(mid, cm.ComplexPair)
    assert mid.real == 4.0
    assert mid.imag == 7.0

    out = cm.ComplexPair()
    res = cm.midpoint_complex(a, b, out_z=out)
    assert res is None
    assert out.real == 4.0
    assert out.imag == 7.0


def test_struct_array_output_returns_array_wrapper(cm):
    pairs = cm.make_complex_pairs()
    assert isinstance(pairs, cm.ComplexPairArray)
    assert repr(pairs) == "<Array ComplexPair[3]>"
    assert pairs.shape == (3,)
    np.testing.assert_allclose(pairs.real, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(pairs.imag, np.array([0.0, -1.0, -2.0]))
    np.testing.assert_allclose(pairs["real"], pairs.real)
    assert isinstance(pairs[1], cm.ComplexPair)
    assert pairs[1].real == 1.0
    assert pairs[1].imag == -1.0


def test_struct_array_initialization(cm):
    from_array_like = cm.ComplexPairArray([(1.0, 2.0), (3.0, 4.0)])
    assert isinstance(from_array_like[0], cm.ComplexPair)
    np.testing.assert_allclose(from_array_like.real, [1.0, 3.0])
    np.testing.assert_allclose(from_array_like.imag, [2.0, 4.0])

    from_fields = cm.ComplexPairArray(real=[5.0, 6.0], imag=-1.0)
    np.testing.assert_allclose(from_fields.real, [5.0, 6.0])
    np.testing.assert_allclose(from_fields.imag, [-1.0, -1.0])


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


def test_owned_array(cm):
    arr, n = cm.alloc_random_array()
    assert arr.shape[0] == n
    assert n >= 3
    assert np.all(arr[:n] >= 1.0)


def test_struct_output_array(cm):
    n = 4
    particles = cm.Particle.zeros(n)
    res = cm.make_particles(3.0, out_p=particles, len_p=n)
    assert res is None
    np.testing.assert_allclose(particles["pos"], np.ones((n, 3)))
    np.testing.assert_allclose(particles["vel"], np.array([[3.0, 0.0, 0.0]] * n))


def test_struct_output_array_auto_return(cm):
    n = 3
    particles = cm.make_particles(2.5, len_p=n)
    np.testing.assert_allclose(particles["pos"], np.ones((n, 3)))
    np.testing.assert_allclose(particles["vel"], np.array([[2.5, 0.0, 0.0]] * n))
