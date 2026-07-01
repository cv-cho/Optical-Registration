from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import SimpleITK as sitk

from .geometry import image_geometry_payload


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SeriesSelection:
    patient_id: str
    modality: str
    series_description: str
    series_uid: str
    study_path: Path
    study_uid: str
    instance_count: int
    slice_thickness: float | None
    image_type: str


@dataclass
class LoadedVolume:
    selection: SeriesSelection
    image: Any
    array_zyx: np.ndarray
    geometry: dict[str, object]


def read_manifest_rows(manifest_csv: str | Path) -> list[dict[str, str]]:
    path = Path(manifest_csv)
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def select_series(
    manifest_csv: str | Path,
    patient_id: str,
    modality: str,
    series_description: str,
) -> SeriesSelection:
    modality = modality.upper()
    matches = [
        row
        for row in read_manifest_rows(manifest_csv)
        if row.get("patient_id") == str(patient_id)
        and row.get("modality", "").upper() == modality
        and row.get("series_description") == series_description
    ]
    if not matches:
        raise RuntimeError(
            f"No matching series: patient={patient_id}, modality={modality}, description={series_description}"
        )
    matches.sort(key=lambda row: int(float(row.get("instance_count") or 0)), reverse=True)
    return selection_from_manifest_row(matches[0], modality)


def select_series_by_uid(
    manifest_csv: str | Path,
    patient_id: str,
    modality: str,
    series_uid: str,
) -> SeriesSelection:
    modality = modality.upper()
    matches = [
        row
        for row in read_manifest_rows(manifest_csv)
        if row.get("patient_id") == str(patient_id)
        and row.get("modality", "").upper() == modality
        and row.get("series_uid") == series_uid
    ]
    if not matches:
        raise RuntimeError(f"No matching series UID: patient={patient_id}, modality={modality}, uid={series_uid}")
    return selection_from_manifest_row(matches[0], modality)


def selection_from_manifest_row(row: dict[str, str], modality: str | None = None) -> SeriesSelection:
    selected_modality = validate_manifest_modality(modality or str(row.get("modality", "")))
    return SeriesSelection(
        patient_id=str(row["patient_id"]),
        modality=selected_modality,
        series_description=str(row["series_description"]),
        series_uid=str(row["series_uid"]),
        study_path=resolve_data_path(row["study_path"]),
        study_uid=str(row.get("study_folder") or ""),
        instance_count=int(float(row.get("instance_count") or 0)),
        slice_thickness=_float_or_none(row.get("slice_thickness")),
        image_type=str(row.get("image_type") or ""),
    )


def validate_manifest_modality(modality: str) -> str:
    value = modality.upper()
    if value == "MRI":
        value = "MR"
    if value not in {"CT", "MR"}:
        raise ValueError(f"Unknown manifest modality: {modality}")
    return value


def selection_from_queue_row(row: dict[str, str], modality: str) -> SeriesSelection:
    modality = modality.upper()
    prefix = "ct" if modality == "CT" else "mri"
    return SeriesSelection(
        patient_id=str(row["patient_id"]),
        modality=modality,
        series_description=str(row[f"{prefix}_series_description"]),
        series_uid=str(row[f"{prefix}_series_uid"]),
        study_path=resolve_data_path(row[f"{prefix}_study_path"]),
        study_uid="",
        instance_count=int(float(row.get(f"{prefix}_slices") or 0)),
        slice_thickness=None,
        image_type="",
    )


def resolve_data_path(path_value: str | Path) -> Path:
    raw = str(path_value).strip()
    if os.name != "nt":
        raw = raw.replace("\\", "/")
        raw = _project_relative_from_windows_absolute(raw) or raw
    path = Path(raw)
    if path.is_absolute():
        return path
    root = Path(os.environ.get("ORBIT_REGISTRATION_ROOT", REPO_ROOT))
    return root / path


def _project_relative_from_windows_absolute(path_value: str) -> str | None:
    if len(path_value) < 3 or path_value[1:3] != ":/":
        return None
    for marker in ("/data/", "/reports/", "/outputs/"):
        index = path_value.find(marker)
        if index >= 0:
            return path_value[index + 1 :]
    return None


def find_series_files(selection: SeriesSelection) -> list[Path]:
    try:
        gdcm_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(selection.study_path),
            selection.series_uid,
        )
        if gdcm_names:
            return [Path(name) for name in gdcm_names]
    except Exception:
        pass

    files: list[tuple[float, int, Path]] = []
    for path in selection.study_path.iterdir():
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if str(ds.get("SeriesInstanceUID", "")) != selection.series_uid:
            continue
        instance = int(getattr(ds, "InstanceNumber", 0) or 0)
        position_key = _slice_position_key(ds)
        files.append((position_key, instance, path))
    if not files:
        raise RuntimeError(f"No DICOM files found for series UID {selection.series_uid}")
    files.sort(key=lambda item: (item[0], item[1], item[2].name))
    return [path for _, _, path in files]


def load_series_volume(selection: SeriesSelection) -> LoadedVolume:
    files = find_series_files(selection)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in files])
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    array = sitk.GetArrayFromImage(image)
    return LoadedVolume(
        selection=selection,
        image=image,
        array_zyx=array,
        geometry=image_geometry_payload(image),
    )


def load_patient_pair(
    manifest_csv: str | Path,
    patient_id: str,
    ct_series_description: str = "AX",
    mri_series_description: str = "T2 COR dixon_(IN W)_in",
) -> tuple[LoadedVolume, LoadedVolume]:
    ct = load_series_volume(select_series(manifest_csv, patient_id, "CT", ct_series_description))
    mri = load_series_volume(select_series(manifest_csv, patient_id, "MR", mri_series_description))
    return ct, mri


def load_patient_pair_by_uid(
    manifest_csv: str | Path,
    patient_id: str,
    ct_series_uid: str,
    mri_series_uid: str,
    queue_row: dict[str, str] | None = None,
) -> tuple[LoadedVolume, LoadedVolume]:
    try:
        ct_selection = select_series_by_uid(manifest_csv, patient_id, "CT", ct_series_uid)
    except RuntimeError:
        if not queue_row:
            raise
        ct_selection = selection_from_queue_row(queue_row, "CT")
    try:
        mri_selection = select_series_by_uid(manifest_csv, patient_id, "MR", mri_series_uid)
    except RuntimeError:
        if not queue_row:
            raise
        mri_selection = selection_from_queue_row(queue_row, "MRI")
    return load_series_volume(ct_selection), load_series_volume(mri_selection)


def ct_window(array: np.ndarray, low: float, high: float) -> np.ndarray:
    arr = array.astype(np.float32)
    return np.clip((arr - float(low)) / max(float(high - low), 1e-6), 0, 1)


def normalize_percentile(array: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    arr = array.astype(np.float32)
    finite = np.isfinite(arr)
    if int(finite.sum()) < 10:
        return np.zeros_like(arr, dtype=np.float32)
    low, high = np.percentile(arr[finite], [low_pct, high_pct])
    return np.clip((arr - low) / max(float(high - low), 1e-6), 0, 1)


def _float_or_none(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _slice_position_key(ds: pydicom.Dataset) -> float:
    try:
        orientation = np.asarray([float(v) for v in ds.ImageOrientationPatient], dtype=float)
        row = orientation[:3]
        col = orientation[3:]
        normal = np.cross(row, col)
        position = np.asarray([float(v) for v in ds.ImagePositionPatient], dtype=float)
        return float(np.dot(position, normal))
    except Exception:
        return float(getattr(ds, "InstanceNumber", 0) or 0)
