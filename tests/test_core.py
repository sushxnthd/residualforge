import numpy as np
import pytest

from residualforge import heat_residual, summarize


def heat_solution(nt=121, nx=161, alpha=0.2):
    t = np.linspace(0, 0.4, nt)
    x = np.linspace(0, 1, nx)
    u = np.exp(-alpha * np.pi**2 * t[:, None]) * np.sin(np.pi * x[None, :])
    return u, x[1] - x[0], t[1] - t[0]


def test_heat_solution_has_small_residual():
    u, dx, dt = heat_solution()
    assert summarize(heat_residual(u, 0.2, dx, dt)).rmse < 3e-4


def test_local_corruption_raises_residual():
    u, dx, dt = heat_solution()
    base = summarize(heat_residual(u, 0.2, dx, dt)).rmse
    v = u.copy(); v[50:55, 70:75] += 0.15
    assert summarize(heat_residual(v, 0.2, dx, dt)).rmse > base * 100


def test_hotspot_is_reported():
    r = np.zeros((4, 5)); r[2, 3] = -7
    assert summarize(r).hotspot == (2, 3)


def test_invalid_spacing_rejected():
    with pytest.raises(ValueError):
        heat_residual(np.ones((4, 4)), 0.2, 0, 0.1)
