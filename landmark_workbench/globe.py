from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SphereFit:
    center_lps: list[float]
    radius_mm: float
    n_points: int
    rms_residual_mm: float
    max_residual_mm: float
    status: str


def fit_sphere_lps(points_lps: list[list[float]] | np.ndarray) -> SphereFit:
    points = np.asarray(points_lps, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Sphere fitting expects an N x 3 point array.")
    if points.shape[0] < 4:
        raise ValueError("At least 4 surface points are required to fit a sphere.")

    a = np.column_stack([points, np.ones(points.shape[0])])
    b = -np.sum(points * points, axis=1)
    coeff, *_ = np.linalg.lstsq(a, b, rcond=None)
    center = -0.5 * coeff[:3]
    radius_sq = float(np.dot(center, center) - coeff[3])
    if radius_sq <= 0:
        raise ValueError("Fitted sphere has a non-positive radius.")
    radius = float(np.sqrt(radius_sq))
    radial = np.linalg.norm(points - center[None, :], axis=1)
    residual = radial - radius
    rms = float(np.sqrt(np.mean(residual * residual)))
    maximum = float(np.max(np.abs(residual)))
    status = "pass" if rms <= 1.5 and maximum <= 3.0 else "warning"
    return SphereFit(
        center_lps=[float(v) for v in center],
        radius_mm=radius,
        n_points=int(points.shape[0]),
        rms_residual_mm=rms,
        max_residual_mm=maximum,
        status=status,
    )


def fit_globe_spheres(points: list[dict[str, Any]]) -> dict[tuple[str, str], SphereFit]:
    grouped: dict[tuple[str, str], list[list[float]]] = {}
    for point in points:
        key = (str(point["modality"]).upper(), str(point["side"]).upper())
        grouped.setdefault(key, []).append([float(v) for v in point["physical_lps_mm"]])
    fits: dict[tuple[str, str], SphereFit] = {}
    for key, values in grouped.items():
        if len(values) < 4:
            continue
        fits[key] = fit_sphere_lps(values)
    return fits
