from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from .globe import SphereFit


@dataclass
class GlobeManualRegistrationResult:
    transform_fixed_to_moving: Any
    transform_moving_to_fixed: Any
    fixed_centers_lps: dict[str, list[float]]
    moving_centers_lps: dict[str, list[float]]
    pitch_deg: float
    scale_xyz: tuple[float, float, float]
    eye_distance_fixed_mm: float
    eye_distance_moving_mm: float


def estimate_globe_manual_initializer(
    fixed_fits: dict[str, SphereFit],
    moving_fits: dict[str, SphereFit],
    pitch_deg: float = 0.0,
    scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> GlobeManualRegistrationResult:
    required = {"L", "R"}
    if not required.issubset(fixed_fits) or not required.issubset(moving_fits):
        raise ValueError("Both CT and MRI need fitted L/R globe spheres.")

    fixed_left = np.asarray(fixed_fits["L"].center_lps, dtype=float)
    fixed_right = np.asarray(fixed_fits["R"].center_lps, dtype=float)
    moving_left = np.asarray(moving_fits["L"].center_lps, dtype=float)
    moving_right = np.asarray(moving_fits["R"].center_lps, dtype=float)
    fixed_mid = (fixed_left + fixed_right) * 0.5
    moving_mid = (moving_left + moving_right) * 0.5
    fixed_axis = fixed_left - fixed_right
    moving_axis = moving_left - moving_right
    fixed_distance = float(np.linalg.norm(fixed_axis))
    moving_distance = float(np.linalg.norm(moving_axis))
    if fixed_distance <= 1.0e-6 or moving_distance <= 1.0e-6:
        raise ValueError("Cannot initialize from degenerate globe centers.")

    align_rotation = rotation_from_vectors(moving_axis, fixed_axis)
    eye_scale = fixed_distance / moving_distance
    initial_matrix = align_rotation * eye_scale
    initial_translation = fixed_mid - initial_matrix @ moving_mid

    pitch_rotation = axis_angle_rotation(fixed_axis, np.deg2rad(float(pitch_deg)))
    scale_matrix = np.diag([float(v) for v in scale_xyz])
    manual_matrix = pitch_rotation @ scale_matrix
    matrix_moving_to_fixed = manual_matrix @ initial_matrix
    translation_moving_to_fixed = fixed_mid + manual_matrix @ (initial_translation - fixed_mid)

    matrix_fixed_to_moving, translation_fixed_to_moving = invert_affine(
        matrix_moving_to_fixed,
        translation_moving_to_fixed,
    )
    return GlobeManualRegistrationResult(
        transform_fixed_to_moving=sitk_affine(matrix_fixed_to_moving, translation_fixed_to_moving),
        transform_moving_to_fixed=sitk_affine(matrix_moving_to_fixed, translation_moving_to_fixed),
        fixed_centers_lps={"L": fixed_fits["L"].center_lps, "R": fixed_fits["R"].center_lps},
        moving_centers_lps={"L": moving_fits["L"].center_lps, "R": moving_fits["R"].center_lps},
        pitch_deg=float(pitch_deg),
        scale_xyz=tuple(float(v) for v in scale_xyz),
        eye_distance_fixed_mm=fixed_distance,
        eye_distance_moving_mm=moving_distance,
    )


def resample_mri_to_ct(
    moving_mri: Any,
    fixed_ct: Any,
    transform_fixed_to_moving: Any,
    default_value: float = 0.0,
) -> Any:
    return sitk.Resample(
        moving_mri,
        fixed_ct,
        transform_fixed_to_moving,
        sitk.sitkLinear,
        float(default_value),
        moving_mri.GetPixelID(),
    )


def save_globe_manual_registration(result: GlobeManualRegistrationResult, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(result.transform_fixed_to_moving, str(out / "T_ct_fixed_to_mri_moving_for_resample.tfm"))
    sitk.WriteTransform(result.transform_moving_to_fixed, str(out / "T_mri_moving_to_ct_fixed_for_overlay.tfm"))
    payload = {
        "fixed_centers_lps": result.fixed_centers_lps,
        "moving_centers_lps": result.moving_centers_lps,
        "pitch_deg": result.pitch_deg,
        "scale_xyz": result.scale_xyz,
        "eye_distance_fixed_mm": result.eye_distance_fixed_mm,
        "eye_distance_moving_mm": result.eye_distance_moving_mm,
    }
    (out / "globe_manual_registration_qc.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    a = normalized(source)
    b = normalized(target)
    cross = np.cross(a, b)
    dot = float(np.dot(a, b))
    if np.linalg.norm(cross) <= 1.0e-8:
        if dot > 0:
            return np.eye(3)
        axis = normalized(np.cross(a, np.array([1.0, 0.0, 0.0])))
        if np.linalg.norm(axis) <= 1.0e-8:
            axis = normalized(np.cross(a, np.array([0.0, 1.0, 0.0])))
        return axis_angle_rotation(axis, np.pi)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / float(np.dot(cross, cross)))


def axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalized(axis)
    x, y, z = axis
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def invert_affine(matrix: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inverse_matrix = np.linalg.inv(matrix)
    inverse_translation = -inverse_matrix @ translation
    return inverse_matrix, inverse_translation


def sitk_affine(matrix: np.ndarray, translation: np.ndarray) -> Any:
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(tuple(float(v) for v in matrix.reshape(-1)))
    transform.SetTranslation(tuple(float(v) for v in translation))
    return transform


def normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm
