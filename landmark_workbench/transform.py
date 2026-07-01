from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from .qc import registration_landmark_records, usable_landmark_pairs


@dataclass
class LandmarkInitializationResult:
    transform_fixed_to_moving: Any
    transform_moving_to_fixed: Any
    labels_used: list[str]
    residuals: list[dict[str, float | str]]
    median_residual_mm: float
    max_residual_mm: float
    status: str


def estimate_rigid_initializer(
    records: list[dict[str, Any]],
    fixed_modality: str = "MRI",
    moving_modality: str = "CT",
) -> LandmarkInitializationResult:
    fixed_modality = fixed_modality.upper()
    moving_modality = moving_modality.upper()
    records = registration_landmark_records(records)
    labels = usable_landmark_pairs(records)
    if len(labels) < 3:
        raise ValueError("At least 3 paired landmarks are required for rigid initialization.")

    by_key = {
        (str(record["modality"]).upper(), str(record["landmark_label"])): record
        for record in records
        if record.get("use_for_transform", True)
    }
    fixed_points = []
    moving_points = []
    labels_used = []
    for label in labels:
        fixed = by_key.get((fixed_modality, label))
        moving = by_key.get((moving_modality, label))
        if not fixed or not moving:
            continue
        fixed_points.extend([float(v) for v in fixed["physical_lps_mm"]])
        moving_points.extend([float(v) for v in moving["physical_lps_mm"]])
        labels_used.append(label)
    if len(labels_used) < 3:
        raise ValueError("At least 3 fixed/moving landmark pairs are required.")

    transform_fixed_to_moving = sitk.LandmarkBasedTransformInitializer(
        sitk.VersorRigid3DTransform(),
        fixed_points,
        moving_points,
    )
    transform_moving_to_fixed = transform_fixed_to_moving.GetInverse()
    residuals = _residuals_moving_to_fixed(by_key, labels_used, fixed_modality, moving_modality, transform_moving_to_fixed)
    values = [float(item["residual_mm"]) for item in residuals]
    median = float(np.median(values))
    maximum = float(np.max(values))
    return LandmarkInitializationResult(
        transform_fixed_to_moving=transform_fixed_to_moving,
        transform_moving_to_fixed=transform_moving_to_fixed,
        labels_used=labels_used,
        residuals=residuals,
        median_residual_mm=median,
        max_residual_mm=maximum,
        status=_qc_status(median, maximum),
    )


def save_initialization_result(result: LandmarkInitializationResult, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(result.transform_fixed_to_moving, str(out / "T_fixed_to_moving_for_resample.tfm"))
    sitk.WriteTransform(result.transform_moving_to_fixed, str(out / "T_moving_to_fixed_for_overlay.tfm"))
    with (out / "landmark_residuals.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "residual_mm"])
        writer.writeheader()
        writer.writerows(result.residuals)
    qc = {
        "n_landmarks_used": len(result.labels_used),
        "labels_used": result.labels_used,
        "median_residual_mm": result.median_residual_mm,
        "max_residual_mm": result.max_residual_mm,
        "status": result.status,
    }
    (out / "registration_init_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")


def result_summary(result: LandmarkInitializationResult) -> dict[str, Any]:
    return {
        "n_landmarks_used": len(result.labels_used),
        "labels_used": result.labels_used,
        "median_residual_mm": result.median_residual_mm,
        "max_residual_mm": result.max_residual_mm,
        "status": result.status,
        "residuals": result.residuals,
    }


def resample_moving_to_fixed(
    moving_image: Any,
    fixed_image: Any,
    transform_fixed_to_moving: Any,
    default_value: float = -1024.0,
    interpolator: int = sitk.sitkLinear,
) -> Any:
    return sitk.Resample(
        moving_image,
        fixed_image,
        transform_fixed_to_moving,
        interpolator,
        float(default_value),
        moving_image.GetPixelID(),
    )


def write_resampled_moving_to_fixed(
    moving_image: Any,
    fixed_image: Any,
    transform_fixed_to_moving: Any,
    output_path: str | Path,
    default_value: float = -1024.0,
) -> Any:
    resampled = resample_moving_to_fixed(
        moving_image=moving_image,
        fixed_image=fixed_image,
        transform_fixed_to_moving=transform_fixed_to_moving,
        default_value=default_value,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled, str(out))
    return resampled


def _residuals_moving_to_fixed(
    by_key: dict[tuple[str, str], dict[str, Any]],
    labels: list[str],
    fixed_modality: str,
    moving_modality: str,
    transform_moving_to_fixed: Any,
) -> list[dict[str, float | str]]:
    residuals: list[dict[str, float | str]] = []
    for label in labels:
        fixed = np.asarray(by_key[(fixed_modality, label)]["physical_lps_mm"], dtype=float)
        moving = tuple(float(v) for v in by_key[(moving_modality, label)]["physical_lps_mm"])
        projected = np.asarray(transform_moving_to_fixed.TransformPoint(moving), dtype=float)
        residuals.append({"label": label, "residual_mm": float(np.linalg.norm(projected - fixed))})
    return residuals


def _qc_status(median_residual_mm: float, max_residual_mm: float) -> str:
    if median_residual_mm <= 3.0 and max_residual_mm <= 5.0:
        return "pass"
    if median_residual_mm <= 4.0 and max_residual_mm <= 7.0:
        return "warning"
    return "fail"
