from __future__ import annotations

from itertools import product
from typing import Any, Iterable

import numpy as np


def napari_zyx_to_itk_xyz(point_zyx: Iterable[float]) -> tuple[float, float, float]:
    z, y, x = [float(v) for v in point_zyx]
    return (x, y, z)


def itk_xyz_to_napari_zyx(index_xyz: Iterable[float]) -> tuple[float, float, float]:
    x, y, z = [float(v) for v in index_xyz]
    return (z, y, x)


def physical_lps_from_napari_zyx(image: Any, point_zyx: Iterable[float]) -> tuple[float, float, float]:
    index_xyz = napari_zyx_to_itk_xyz(point_zyx)
    return tuple(float(v) for v in image.TransformContinuousIndexToPhysicalPoint(index_xyz))


def napari_zyx_from_physical_lps(image: Any, point_lps: Iterable[float]) -> tuple[float, float, float]:
    index_xyz = image.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in point_lps))
    return itk_xyz_to_napari_zyx(index_xyz)


def image_direction_3x3(image: Any) -> list[list[float]]:
    direction = [float(v) for v in image.GetDirection()]
    return [direction[0:3], direction[3:6], direction[6:9]]


def image_geometry_payload(image: Any) -> dict[str, object]:
    return {
        "size_xyz": [int(v) for v in image.GetSize()],
        "spacing_xyz": [float(v) for v in image.GetSpacing()],
        "origin_lps": [float(v) for v in image.GetOrigin()],
        "direction_3x3": image_direction_3x3(image),
        "array_shape_zyx": [int(v) for v in reversed(image.GetSize())],
        "physical_bounding_box_lps": image_physical_bounding_box_lps(image),
    }


def image_physical_bounding_box_lps(image: Any) -> dict[str, list[float]]:
    size = image.GetSize()
    if any(int(v) <= 0 for v in size):
        raise ValueError("Cannot calculate a bounding box for an empty image.")
    corners = []
    for corner in product(*[(0, int(dim) - 1) for dim in size]):
        corners.append(image.TransformIndexToPhysicalPoint(tuple(int(v) for v in corner)))
    arr = np.asarray(corners, dtype=float)
    return {"min": arr.min(axis=0).tolist(), "max": arr.max(axis=0).tolist()}


def point_inside_physical_box(image: Any, point_lps: Iterable[float], tolerance_mm: float = 1e-3) -> bool:
    box = image_physical_bounding_box_lps(image)
    point = np.asarray(tuple(float(v) for v in point_lps), dtype=float)
    low = np.asarray(box["min"], dtype=float) - float(tolerance_mm)
    high = np.asarray(box["max"], dtype=float) + float(tolerance_mm)
    return bool(np.all(point >= low) and np.all(point <= high))


def point_inside_array_shape_zyx(point_zyx: Iterable[float], shape_zyx: Iterable[int]) -> bool:
    point = np.asarray(tuple(float(v) for v in point_zyx), dtype=float)
    shape = np.asarray(tuple(int(v) for v in shape_zyx), dtype=float)
    if point.shape != (3,) or shape.shape != (3,):
        return False
    return bool(np.all(point >= 0.0) and np.all(point <= (shape - 1.0)))
