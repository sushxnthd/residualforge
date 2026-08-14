from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ResidualSummary:
    rmse: float
    p95: float
    max_abs: float
    hotspot: tuple[int, int]


def _validate(u: np.ndarray, dx: float, dt: float) -> np.ndarray:
    x = np.asarray(u, dtype=float)
    if x.ndim != 2 or min(x.shape) < 3:
        raise ValueError("u must be a 2D time-by-space array with at least 3 points per axis")
    if dx <= 0 or dt <= 0:
        raise ValueError("dx and dt must be positive")
    return x


def derivatives(u: np.ndarray, dx: float, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _validate(u, dx, dt)
    ut = (x[2:, 1:-1] - x[:-2, 1:-1]) / (2.0 * dt)
    ux = (x[1:-1, 2:] - x[1:-1, :-2]) / (2.0 * dx)
    uxx = (x[1:-1, 2:] - 2.0 * x[1:-1, 1:-1] + x[1:-1, :-2]) / (dx * dx)
    return ut, ux, uxx


def heat_residual(u: np.ndarray, alpha: float, dx: float, dt: float) -> np.ndarray:
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    ut, _, uxx = derivatives(u, dx, dt)
    return ut - alpha * uxx


def burgers_residual(u: np.ndarray, nu: float, dx: float, dt: float) -> np.ndarray:
    if nu < 0:
        raise ValueError("nu must be non-negative")
    x = _validate(u, dx, dt)
    ut, ux, uxx = derivatives(x, dx, dt)
    return ut + x[1:-1, 1:-1] * ux - nu * uxx


def summarize(residual: np.ndarray) -> ResidualSummary:
    r = np.asarray(residual, dtype=float)
    if r.size == 0 or not np.isfinite(r).all():
        raise ValueError("residual must contain finite values")
    a = np.abs(r)
    idx = np.unravel_index(int(np.argmax(a)), a.shape)
    return ResidualSummary(
        rmse=float(np.sqrt(np.mean(r * r))),
        p95=float(np.quantile(a, 0.95)),
        max_abs=float(np.max(a)),
        hotspot=(int(idx[0]), int(idx[1])),
    )
