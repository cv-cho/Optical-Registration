from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR))

from landmark_workbench.globe import SphereFit, fit_globe_spheres  # noqa: E402
from landmark_workbench.globe_registration import (  # noqa: E402
    estimate_globe_manual_initializer,
    save_globe_manual_registration,
)
from landmark_workbench.store import AnnotationStore  # noqa: E402
from landmark_workbench.volume import load_patient_pair_by_uid  # noqa: E402


DEFAULT_SERIES = "T2 COR dixon_(IN W)_in"
DEFAULT_OUTPUT_NAME = "T2_COR_dixon_IN_W_in_registered_brightness_trimmed"
DEFAULT_EXCLUDED_PATIENTS = {"102697"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export completed T2 COR CT/MRI registration cases into MRI-grid paired training volumes. "
            "CT is resampled to the native MRI grid, cropped to common CT/MRI support, and then "
            "patient-wise low-brightness edge slices are trimmed with the existing project rule."
        )
    )
    parser.add_argument("--manifest", type=Path, default=WORKDIR / "reports" / "series_inventory_series.csv")
    parser.add_argument("--work-queue", type=Path, default=WORKDIR / "outputs" / "landmarks" / "work_queue.csv")
    parser.add_argument("--db", type=Path, default=WORKDIR / "outputs" / "landmarks" / "annotations.sqlite")
    parser.add_argument("--mri-series", default=DEFAULT_SERIES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKDIR / "outputs" / "training_pairs" / DEFAULT_OUTPUT_NAME,
    )
    parser.add_argument("--patients", nargs="*", default=None, help="Optional patient id subset.")
    parser.add_argument(
        "--exclude-patient",
        action="append",
        default=[],
        help="Patient id to exclude from export. Can be passed more than once.",
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Do not automatically exclude the known data-problem patient 102697.",
    )
    parser.add_argument("--ct-clip-low", type=float, default=-1000.0)
    parser.add_argument("--ct-clip-high", type=float, default=2000.0)
    parser.add_argument("--mri-low-pct", type=float, default=0.5)
    parser.add_argument("--mri-high-pct", type=float, default=99.5)
    parser.add_argument("--min-points-per-eye", type=int, default=4)
    parser.add_argument("--crop-margin-voxels", type=int, default=8)
    parser.add_argument("--no-crop", action="store_true", help="Keep the full native MRI grid.")
    parser.add_argument("--foreground-threshold", type=int, default=5)
    parser.add_argument("--mad-multiplier", type=float, default=3.0)
    parser.add_argument("--min-fraction-drop", type=float, default=0.30)
    parser.add_argument("--relative-min-fraction", type=float, default=0.55)
    parser.add_argument(
        "--min-kept-slices",
        type=int,
        default=1,
        help=(
            "Safety guard after brightness trimming. The documented trim rule has no fixed minimum; "
            "default is 1 because T2 COR volumes commonly have 30 slices."
        ),
    )
    parser.add_argument("--no-brightness-trim", action="store_true")
    parser.add_argument(
        "--save-raw-like",
        action="store_true",
        help="Also save CT HU on MRI grid and native MRI intensity volumes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory before exporting.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only list cases that would be exported.")
    parser.add_argument(
        "--make-archive",
        action="store_true",
        help="Create a tar.gz archive of the exported dataset after a successful export.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=None,
        help="Archive path used with --make-archive. Defaults to <output-dir>.tar.gz.",
    )
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"Missing annotation database: {args.db}")
    if not args.manifest.exists():
        raise SystemExit(f"Missing series manifest: {args.manifest}")

    requested_patients = {str(v) for v in args.patients} if args.patients else None
    excluded_patients = set(str(v) for v in args.exclude_patient)
    if not args.no_default_excludes:
        excluded_patients.update(DEFAULT_EXCLUDED_PATIENTS)

    queue_rows = read_queue_rows(args.work_queue)
    all_cases = discover_cases(args.db, args.mri_series)
    cases, excluded_rows = filter_cases(
        all_cases,
        requested_patients=requested_patients,
        excluded_patients=excluded_patients,
    )

    if args.dry_run:
        summary = {
            "dry_run": True,
            "mri_series": args.mri_series,
            "annotation_db": str(args.db),
            "registered_cases_in_db": len(all_cases),
            "selected_cases": len(cases),
            "excluded_cases": excluded_rows,
            "selected_patient_ids": [case["patient_id"] for case in cases],
            "output_dir": str(args.output_dir),
            "brightness_trim_enabled": not bool(args.no_brightness_trim),
            "default_excluded_patients": sorted(DEFAULT_EXCLUDED_PATIENTS),
        }
        print(json.dumps(summary, indent=2), flush=True)
        return

    output_dir = args.output_dir
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    create_output_dirs(output_dir)

    store = AnnotationStore(args.db)
    exported_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = list(excluded_rows)
    brightness_report_rows: list[dict[str, Any]] = []
    slice_report_rows: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(cases, start=1):
            patient_id = str(case["patient_id"])
            print(f"[{index}/{len(cases)}] {patient_id}", flush=True)
            try:
                if not args.overwrite and case_export_complete(output_dir, patient_id):
                    row = manifest_row_from_metadata(output_dir / "metadata" / f"{safe_path_component(patient_id)}.json")
                    exported_rows.append(row)
                    print("  already complete; skipped", flush=True)
                    continue
                row, brightness_row, case_slice_rows = export_case(
                    case=case,
                    manifest_csv=args.manifest,
                    queue_rows=queue_rows,
                    store=store,
                    output_dir=output_dir,
                    ct_clip_low=float(args.ct_clip_low),
                    ct_clip_high=float(args.ct_clip_high),
                    mri_low_pct=float(args.mri_low_pct),
                    mri_high_pct=float(args.mri_high_pct),
                    min_points_per_eye=int(args.min_points_per_eye),
                    save_raw_like=bool(args.save_raw_like),
                    crop_to_overlap=not bool(args.no_crop),
                    crop_margin_voxels=int(args.crop_margin_voxels),
                    brightness_trim_enabled=not bool(args.no_brightness_trim),
                    foreground_threshold=int(args.foreground_threshold),
                    mad_multiplier=float(args.mad_multiplier),
                    min_fraction_drop=float(args.min_fraction_drop),
                    relative_min_fraction=float(args.relative_min_fraction),
                    min_kept_slices=int(args.min_kept_slices),
                )
                exported_rows.append(row)
                if brightness_row:
                    brightness_report_rows.append(brightness_row)
                slice_report_rows.extend(case_slice_rows)
                print(
                    f"  exported: shape={row['shape_zyx']} ct_valid_fraction={float(row['ct_valid_fraction']):.4f}",
                    flush=True,
                )
            except Exception as exc:
                skipped = {"patient_id": patient_id, "reason": str(exc)}
                skipped_rows.append(skipped)
                print(f"  skipped: {exc}", flush=True)
    finally:
        store.close()

    all_exported_rows = manifest_rows_from_metadata_dir(output_dir / "metadata")
    write_csv(output_dir / "dataset_manifest.csv", all_exported_rows)
    write_csv(output_dir / "skipped_cases.csv", skipped_rows)
    write_csv(output_dir / "brightness_trim_report.csv", brightness_report_rows)
    write_csv(output_dir / "reports" / "brightness_trim_slice_report.csv", slice_report_rows)

    summary = build_summary(
        args=args,
        output_dir=output_dir,
        all_cases=all_cases,
        exported_rows=all_exported_rows,
        skipped_rows=skipped_rows,
        excluded_patients=excluded_patients,
        brightness_report_rows=brightness_report_rows,
    )
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.make_archive:
        archive_path = args.archive_path or output_dir.with_suffix(".tar.gz")
        create_archive(output_dir, archive_path)
        summary["archive_path"] = str(archive_path)
        (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)


def discover_cases(db_path: Path, mri_series: str) -> list[dict[str, Any]]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT patient_id, ct_series_uid, mri_series_uid, mri_series_description,
                   pitch_deg, scale_xyz, updated_at
            FROM globe_manual_parameters
            WHERE mri_series_description=?
            ORDER BY CAST(patient_id AS INTEGER), patient_id
            """,
            (mri_series,),
        ).fetchall()
    finally:
        con.close()
    cases: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["patient_id"] = str(item["patient_id"])
        item["scale_xyz"] = json.loads(item["scale_xyz"]) if isinstance(item["scale_xyz"], str) else item["scale_xyz"]
        cases.append(item)
    return cases


def filter_cases(
    cases: list[dict[str, Any]],
    requested_patients: set[str] | None,
    excluded_patients: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for case in cases:
        patient_id = str(case["patient_id"])
        if requested_patients and patient_id not in requested_patients:
            continue
        if patient_id in excluded_patients:
            excluded_rows.append({"patient_id": patient_id, "reason": "excluded_patient_data_issue"})
            continue
        selected.append(case)
    return selected, excluded_rows


def export_case(
    case: dict[str, Any],
    manifest_csv: Path,
    queue_rows: dict[str, dict[str, str]],
    store: AnnotationStore,
    output_dir: Path,
    ct_clip_low: float,
    ct_clip_high: float,
    mri_low_pct: float,
    mri_high_pct: float,
    min_points_per_eye: int,
    save_raw_like: bool,
    crop_to_overlap: bool,
    crop_margin_voxels: int,
    brightness_trim_enabled: bool,
    foreground_threshold: int,
    mad_multiplier: float,
    min_fraction_drop: float,
    relative_min_fraction: float,
    min_kept_slices: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
    patient_id = str(case["patient_id"])
    ct_uid = str(case["ct_series_uid"])
    mri_uid = str(case["mri_series_uid"])
    ct_volume, mri_volume = load_patient_pair_by_uid(
        manifest_csv,
        patient_id,
        ct_uid,
        mri_uid,
        queue_row=queue_rows.get(patient_id),
    )

    points = []
    points.extend(store.fetch_globe_surface_points(patient_id, "CT", series_uid=ct_uid))
    points.extend(store.fetch_globe_surface_points(patient_id, "MRI", series_uid=mri_uid))
    counts = point_counts(points)
    fits = fit_globe_spheres(points)
    center_overrides = fetch_center_overrides(store, patient_id, ct_uid=ct_uid, mri_uid=mri_uid)
    fits, override_keys = apply_center_overrides(fits, center_overrides)
    validate_required_globes(fits, counts, override_keys, min_points_per_eye)

    fixed_fits = {side: fits[("CT", side)] for side in ("L", "R")}
    moving_fits = {side: fits[("MRI", side)] for side in ("L", "R")}
    pitch = float(case["pitch_deg"])
    scale_xyz = tuple(float(v) for v in case["scale_xyz"])
    result = estimate_globe_manual_initializer(
        fixed_fits=fixed_fits,
        moving_fits=moving_fits,
        pitch_deg=pitch,
        scale_xyz=scale_xyz,
    )

    case_prefix = safe_path_component(patient_id)
    transform_dir = output_dir / "transforms" / case_prefix
    transform_dir.mkdir(parents=True, exist_ok=True)
    save_globe_manual_registration(result, transform_dir)

    ct_on_mri = resample_ct_to_mri_grid(ct_volume.image, mri_volume.image, result.transform_moving_to_fixed)
    mri_native = sitk.Cast(mri_volume.image, sitk.sitkFloat32)
    ct_valid_mask_on_mri = resample_ct_valid_mask_to_mri_grid(
        ct_volume.image,
        mri_volume.image,
        result.transform_moving_to_fixed,
    )

    ct_norm = image_from_array_like(
        scale_ct_to_minus_one_one(sitk.GetArrayFromImage(ct_on_mri), ct_clip_low, ct_clip_high),
        ct_on_mri,
        sitk.sitkFloat32,
    )
    mri_norm, mri_low, mri_high = normalize_mri_to_minus_one_one(
        mri_native,
        low_pct=mri_low_pct,
        high_pct=mri_high_pct,
    )

    full_shape_zyx = list(sitk.GetArrayFromImage(mri_volume.image).shape)
    crop_bbox_zyx: list[list[int]] | None = None
    crop_source = "none"
    if crop_to_overlap:
        mask_array_full = sitk.GetArrayFromImage(ct_valid_mask_on_mri) > 0
        mri_norm_array_full = sitk.GetArrayFromImage(mri_norm)
        overlap = mask_array_full & (mri_norm_array_full > -0.98)
        if int(overlap.sum()) >= 100:
            crop_bbox_zyx = bbox_zyx(overlap, margin=int(crop_margin_voxels), shape_zyx=overlap.shape)
            crop_source = "ct_valid_mask_and_mri_foreground"
        elif int(mask_array_full.sum()) >= 100:
            crop_bbox_zyx = bbox_zyx(mask_array_full, margin=int(crop_margin_voxels), shape_zyx=mask_array_full.shape)
            crop_source = "ct_valid_mask"
        if crop_bbox_zyx is not None:
            ct_on_mri = crop_image_zyx(ct_on_mri, crop_bbox_zyx)
            mri_native = crop_image_zyx(mri_native, crop_bbox_zyx)
            ct_valid_mask_on_mri = crop_image_zyx(ct_valid_mask_on_mri, crop_bbox_zyx)
            ct_norm = crop_image_zyx(ct_norm, crop_bbox_zyx)
            mri_norm = crop_image_zyx(mri_norm, crop_bbox_zyx)

    pre_brightness_trim_shape_zyx = list(sitk.GetArrayFromImage(mri_norm).shape)
    brightness_payload: dict[str, Any] | None = None
    brightness_row: dict[str, Any] | None = None
    slice_report_rows: list[dict[str, Any]] = []
    if brightness_trim_enabled:
        planned = plan_brightness_trim(
            patient_id=patient_id,
            mri_norm=mri_norm,
            mask=ct_valid_mask_on_mri,
            foreground_threshold=foreground_threshold,
            mad_multiplier=mad_multiplier,
            min_fraction_drop=min_fraction_drop,
            relative_min_fraction=relative_min_fraction,
            min_kept_slices=min_kept_slices,
        )
        z_start = int(planned["z_start"])
        z_end = int(planned["z_end"])
        ct_on_mri = crop_z(ct_on_mri, z_start, z_end)
        mri_native = crop_z(mri_native, z_start, z_end)
        ct_valid_mask_on_mri = crop_z(ct_valid_mask_on_mri, z_start, z_end)
        ct_norm = crop_z(ct_norm, z_start, z_end)
        mri_norm = crop_z(mri_norm, z_start, z_end)
        brightness_payload = brightness_metadata(planned)
        brightness_row = strip_slice_detail(planned)
        slice_report_rows = list(planned["slice_report_rows"])

    paths = {
        "ct_on_mri_norm": output_dir / "images" / f"{case_prefix}_ct_on_mri_norm.nii",
        "mri_norm": output_dir / "images" / f"{case_prefix}_mri_norm.nii",
        "ct_valid_mask_on_mri": output_dir / "masks" / f"{case_prefix}_ct_valid_mask_on_mri.nii",
    }
    if save_raw_like:
        paths["ct_on_mri_hu"] = output_dir / "images" / f"{case_prefix}_ct_on_mri_hu.nii"
        paths["mri_native"] = output_dir / "images" / f"{case_prefix}_mri_native.nii"
        sitk.WriteImage(ct_on_mri, str(paths["ct_on_mri_hu"]))
        sitk.WriteImage(mri_native, str(paths["mri_native"]))
    sitk.WriteImage(ct_norm, str(paths["ct_on_mri_norm"]))
    sitk.WriteImage(mri_norm, str(paths["mri_norm"]))
    sitk.WriteImage(ct_valid_mask_on_mri, str(paths["ct_valid_mask_on_mri"]))

    mask_array = sitk.GetArrayFromImage(ct_valid_mask_on_mri)
    valid_fraction = float(np.count_nonzero(mask_array) / max(mask_array.size, 1))
    output_shape_zyx = list(sitk.GetArrayFromImage(mri_norm).shape)

    metadata = {
        "patient_id": patient_id,
        "ct_series_uid": ct_uid,
        "mri_series_uid": mri_uid,
        "mri_series_description": str(case["mri_series_description"]),
        "grid": {
            "reference": "MRI",
            "crop_source": crop_source,
            "crop_bbox_zyx": crop_bbox_zyx,
            "full_shape_zyx": full_shape_zyx,
            "pre_brightness_trim_shape_zyx": pre_brightness_trim_shape_zyx,
            "brightness_trim_applied": bool(brightness_payload),
            "shape_zyx": output_shape_zyx,
            "size_xyz": list(mri_norm.GetSize()),
            "spacing_xyz": [float(v) for v in mri_norm.GetSpacing()],
            "origin_lps": [float(v) for v in mri_norm.GetOrigin()],
            "direction_3x3": matrix_3x3(mri_norm.GetDirection()),
        },
        "manual_registration": {
            "pitch_deg": pitch,
            "scale_xyz": list(scale_xyz),
            "eye_distance_ct_mm": result.eye_distance_fixed_mm,
            "eye_distance_mri_mm": result.eye_distance_moving_mm,
            "point_counts": stringify_count_keys(counts),
            "center_overrides": center_overrides,
            "updated_at": case.get("updated_at"),
        },
        "normalization": {
            "ct_clip_hu": [ct_clip_low, ct_clip_high],
            "mri_percentiles": [mri_low, mri_high],
        },
        "brightness_trim": brightness_payload,
        "ct_valid_fraction": valid_fraction,
        "paths": {name: str(path) for name, path in paths.items()},
        "transform_dir": str(transform_dir),
    }
    metadata_path = output_dir / "metadata" / f"{case_prefix}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return manifest_row_from_metadata(metadata_path), brightness_row, slice_report_rows


def resample_ct_to_mri_grid(ct_image: sitk.Image, mri_image: sitk.Image, transform_mri_to_ct: sitk.Transform) -> sitk.Image:
    return sitk.Resample(
        ct_image,
        mri_image,
        transform_mri_to_ct,
        sitk.sitkLinear,
        -1024.0,
        sitk.sitkFloat32,
    )


def resample_ct_valid_mask_to_mri_grid(
    ct_image: sitk.Image,
    mri_image: sitk.Image,
    transform_mri_to_ct: sitk.Transform,
) -> sitk.Image:
    ct_mask = sitk.Image(ct_image.GetSize(), sitk.sitkUInt8)
    ct_mask.CopyInformation(ct_image)
    ct_mask = ct_mask + 1
    return sitk.Resample(
        ct_mask,
        mri_image,
        transform_mri_to_ct,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def point_counts(points: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for point in points:
        key = (str(point["modality"]).upper(), str(point["side"]).upper())
        counts[key] = counts.get(key, 0) + 1
    return counts


def fetch_center_overrides(
    store: AnnotationStore,
    patient_id: str,
    ct_uid: str,
    mri_uid: str,
) -> list[dict[str, Any]]:
    overrides: list[dict[str, Any]] = []
    for modality, series_uid in (("CT", ct_uid), ("MRI", mri_uid)):
        for row in store.fetch_globe_center_overrides(patient_id, modality, series_uid=series_uid):
            overrides.append(
                {
                    "modality": str(row["modality"]).upper(),
                    "side": str(row["side"]).upper(),
                    "series_uid": str(row["series_uid"]),
                    "physical_lps_mm": [float(v) for v in row["physical_lps_mm"]],
                    "source": str(row.get("source") or "manual_globe_center_override"),
                    "updated_at": str(row.get("updated_at") or ""),
                }
            )
    return overrides


def apply_center_overrides(
    fits: dict[tuple[str, str], SphereFit],
    center_overrides: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], SphereFit], set[tuple[str, str]]]:
    adjusted = dict(fits)
    override_keys: set[tuple[str, str]] = set()
    for row in center_overrides:
        key = (str(row["modality"]).upper(), str(row["side"]).upper())
        existing = adjusted.get(key)
        adjusted[key] = sphere_fit_with_replaced_center(existing, row["physical_lps_mm"])
        override_keys.add(key)
    return adjusted, override_keys


def sphere_fit_with_replaced_center(existing: SphereFit | None, center_lps: list[float]) -> SphereFit:
    if existing is None:
        return SphereFit(
            center_lps=[float(v) for v in center_lps],
            radius_mm=12.0,
            n_points=0,
            rms_residual_mm=0.0,
            max_residual_mm=0.0,
            status="manual_center_override",
        )
    return SphereFit(
        center_lps=[float(v) for v in center_lps],
        radius_mm=float(existing.radius_mm),
        n_points=int(existing.n_points),
        rms_residual_mm=float(existing.rms_residual_mm),
        max_residual_mm=float(existing.max_residual_mm),
        status=f"{existing.status}_manual_center_override",
    )


def validate_required_globes(
    fits: dict[tuple[str, str], SphereFit],
    counts: dict[tuple[str, str], int],
    override_keys: set[tuple[str, str]],
    min_points_per_eye: int,
) -> None:
    for key in [("CT", "L"), ("CT", "R"), ("MRI", "L"), ("MRI", "R")]:
        if key not in fits:
            raise RuntimeError(f"missing fitted or forced globe center for {key}")
        if key not in override_keys and counts.get(key, 0) < int(min_points_per_eye):
            raise RuntimeError(f"not enough globe surface points for {key}: {counts.get(key, 0)}")


def scale_ct_to_minus_one_one(array: np.ndarray, low: float, high: float) -> np.ndarray:
    arr = array.astype(np.float32)
    scaled = (np.clip(arr, low, high) - low) / max(high - low, 1.0e-6)
    return (scaled * 2.0 - 1.0).astype(np.float32)


def normalize_mri_to_minus_one_one(image: sitk.Image, low_pct: float, high_pct: float) -> tuple[sitk.Image, float, float]:
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    finite = np.isfinite(arr)
    mask = finite & (arr != 0)
    if int(mask.sum()) < 10:
        mask = finite
    if int(mask.sum()) < 10:
        norm = np.zeros_like(arr, dtype=np.float32)
        return image_from_array_like(norm, image, sitk.sitkFloat32), 0.0, 1.0
    low, high = np.percentile(arr[mask], [low_pct, high_pct])
    unit = np.clip((arr - low) / max(float(high - low), 1.0e-6), 0.0, 1.0)
    norm = (unit * 2.0 - 1.0).astype(np.float32)
    norm[~finite] = -1.0
    return image_from_array_like(norm, image, sitk.sitkFloat32), float(low), float(high)


def image_from_array_like(array_zyx: np.ndarray, reference: sitk.Image, pixel_id: int) -> sitk.Image:
    image = sitk.GetImageFromArray(array_zyx.astype(np.float32))
    image.CopyInformation(reference)
    return sitk.Cast(image, pixel_id)


def bbox_zyx(mask: np.ndarray, margin: int, shape_zyx: tuple[int, int, int]) -> list[list[int]]:
    coords = np.argwhere(mask)
    low = coords.min(axis=0)
    high = coords.max(axis=0) + 1
    low = np.maximum(low - int(margin), 0)
    high = np.minimum(high + int(margin), np.asarray(shape_zyx))
    return [[int(v) for v in low], [int(v) for v in high]]


def crop_image_zyx(image: sitk.Image, bbox: list[list[int]]) -> sitk.Image:
    low_zyx, high_zyx = bbox
    size_zyx = [int(hi - lo) for lo, hi in zip(low_zyx, high_zyx)]
    index_xyz = [int(low_zyx[2]), int(low_zyx[1]), int(low_zyx[0])]
    size_xyz = [int(size_zyx[2]), int(size_zyx[1]), int(size_zyx[0])]
    return sitk.RegionOfInterest(image, size_xyz, index_xyz)


def crop_z(image: sitk.Image, z_start: int, z_end: int) -> sitk.Image:
    size = list(image.GetSize())
    index = [0, 0, int(z_start)]
    size[2] = int(z_end - z_start)
    return sitk.RegionOfInterest(image, size, index)


def plan_brightness_trim(
    patient_id: str,
    mri_norm: sitk.Image,
    mask: sitk.Image,
    foreground_threshold: int,
    mad_multiplier: float,
    min_fraction_drop: float,
    relative_min_fraction: float,
    min_kept_slices: int,
) -> dict[str, Any]:
    mri_arr = sitk.GetArrayFromImage(mri_norm).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(mask) > 0
    if mri_arr.shape != mask_arr.shape:
        raise RuntimeError(f"Shape mismatch for {patient_id}: MRI={mri_arr.shape}, mask={mask_arr.shape}")
    if int(mri_arr.shape[0]) < 1:
        raise RuntimeError(f"Empty volume for {patient_id}")

    values = np.array(
        [
            slice_mri_png_like_foreground_mean(mri_arr[z], mask_arr[z], foreground_threshold)
            for z in range(mri_arr.shape[0])
        ],
        dtype=np.float32,
    )
    finite_values = values[np.isfinite(values)]
    if finite_values.size < 3:
        raise RuntimeError(f"Too few valid slices for brightness trim: {patient_id}")

    median = float(np.median(finite_values))
    mad = float(np.median(np.abs(finite_values - median)))
    robust_low_threshold = median - max(float(mad_multiplier) * mad, float(min_fraction_drop) * median)
    relative_low_threshold = float(relative_min_fraction) * median
    low_threshold = max(robust_low_threshold, relative_low_threshold)
    outlier_flags = np.isfinite(values) & (values < low_threshold)

    leading_removed = 0
    while leading_removed < len(outlier_flags) and bool(outlier_flags[leading_removed]):
        leading_removed += 1

    trailing_removed = 0
    while (
        trailing_removed < len(outlier_flags) - leading_removed
        and bool(outlier_flags[len(outlier_flags) - 1 - trailing_removed])
    ):
        trailing_removed += 1

    z_start = int(leading_removed)
    z_end = int(len(outlier_flags) - trailing_removed)
    if z_end - z_start < int(min_kept_slices):
        raise RuntimeError(
            f"Brightness trim would leave too few slices for {patient_id}: "
            f"{z_end - z_start} < {min_kept_slices}"
        )

    slice_report_rows = []
    for z, value in enumerate(values):
        action = "trim" if z < z_start or z >= z_end else "keep"
        if bool(outlier_flags[z]) and action == "keep":
            action = "keep_interior_outlier"
        slice_report_rows.append(
            {
                "patient_id": patient_id,
                "slice_index": int(z),
                "mri_foreground_mean_png_like": float(value),
                "patient_median": median,
                "patient_mad": mad,
                "robust_low_threshold": robust_low_threshold,
                "relative_low_threshold": relative_low_threshold,
                "low_threshold": low_threshold,
                "is_low_outlier": bool(outlier_flags[z]),
                "action": action,
            }
        )

    return {
        "patient_id": patient_id,
        "z_start": z_start,
        "z_end": z_end,
        "source_shape_zyx": list(mri_arr.shape),
        "trimmed_shape_zyx": [int(z_end - z_start), int(mri_arr.shape[1]), int(mri_arr.shape[2])],
        "leading_removed": int(leading_removed),
        "trailing_removed": int(trailing_removed),
        "total_removed": int(leading_removed + trailing_removed),
        "patient_median": median,
        "patient_mad": mad,
        "robust_low_threshold": robust_low_threshold,
        "relative_low_threshold": relative_low_threshold,
        "low_threshold": low_threshold,
        "foreground_threshold_png_like": int(foreground_threshold),
        "mad_multiplier": float(mad_multiplier),
        "min_fraction_drop": float(min_fraction_drop),
        "relative_min_fraction": float(relative_min_fraction),
        "min_kept_slices": int(min_kept_slices),
        "slice_report_rows": slice_report_rows,
    }


def slice_mri_png_like_foreground_mean(slice_2d: np.ndarray, mask_2d: np.ndarray, foreground_threshold: int) -> float:
    scaled = np.clip((slice_2d.astype(np.float32) + 1.0) * 0.5, 0.0, 1.0)
    uint8_like = np.rint(scaled * 255.0).astype(np.uint8)
    foreground = mask_2d & (uint8_like > int(foreground_threshold))
    if int(foreground.sum()) > 0:
        return float(uint8_like[foreground].mean())
    fallback = uint8_like[mask_2d]
    if fallback.size:
        return float(fallback.mean())
    return float(uint8_like.mean())


def brightness_metadata(planned: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": "patient-wise contiguous z-end low-brightness trim, matching the existing 2D PNG filter rule",
        "z_start_in_pre_trim_volume": planned["z_start"],
        "z_end_in_pre_trim_volume_exclusive": planned["z_end"],
        "leading_removed": planned["leading_removed"],
        "trailing_removed": planned["trailing_removed"],
        "total_removed": planned["total_removed"],
        "source_shape_zyx": planned["source_shape_zyx"],
        "trimmed_shape_zyx": planned["trimmed_shape_zyx"],
        "foreground_threshold_png_like": planned["foreground_threshold_png_like"],
        "mad_multiplier": planned["mad_multiplier"],
        "min_fraction_drop": planned["min_fraction_drop"],
        "relative_min_fraction": planned["relative_min_fraction"],
        "min_kept_slices": planned["min_kept_slices"],
        "patient_median": planned["patient_median"],
        "patient_mad": planned["patient_mad"],
        "robust_low_threshold": planned["robust_low_threshold"],
        "relative_low_threshold": planned["relative_low_threshold"],
        "low_threshold": planned["low_threshold"],
    }


def strip_slice_detail(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "patient_id": row["patient_id"],
        "source_shape_zyx": json.dumps(row["source_shape_zyx"], ensure_ascii=True),
        "trimmed_shape_zyx": json.dumps(row["trimmed_shape_zyx"], ensure_ascii=True),
        "z_start": row["z_start"],
        "z_end": row["z_end"],
        "leading_removed": row["leading_removed"],
        "trailing_removed": row["trailing_removed"],
        "total_removed": row["total_removed"],
        "patient_median": row["patient_median"],
        "patient_mad": row["patient_mad"],
        "robust_low_threshold": row["robust_low_threshold"],
        "relative_low_threshold": row["relative_low_threshold"],
        "low_threshold": row["low_threshold"],
        "foreground_threshold_png_like": row["foreground_threshold_png_like"],
        "mad_multiplier": row["mad_multiplier"],
        "min_fraction_drop": row["min_fraction_drop"],
        "relative_min_fraction": row["relative_min_fraction"],
        "min_kept_slices": row["min_kept_slices"],
    }


def stringify_count_keys(counts: dict[tuple[str, str], int]) -> dict[str, int]:
    return {f"{modality}_{side}": int(count) for (modality, side), count in sorted(counts.items())}


def matrix_3x3(direction: tuple[float, ...]) -> list[list[float]]:
    values = [float(v) for v in direction]
    return [values[0:3], values[3:6], values[6:9]]


def read_queue_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {str(row.get("patient_id", "")).strip(): dict(row) for row in csv.DictReader(f)}


def create_output_dirs(output_dir: Path) -> None:
    for subdir in ("images", "masks", "metadata", "transforms", "reports"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def case_export_complete(output_dir: Path, patient_id: str) -> bool:
    case_prefix = safe_path_component(patient_id)
    required = [
        output_dir / "images" / f"{case_prefix}_ct_on_mri_norm.nii",
        output_dir / "images" / f"{case_prefix}_mri_norm.nii",
        output_dir / "masks" / f"{case_prefix}_ct_valid_mask_on_mri.nii",
        output_dir / "metadata" / f"{case_prefix}.json",
        output_dir / "transforms" / case_prefix / "globe_manual_registration_qc.json",
        output_dir / "transforms" / case_prefix / "T_ct_fixed_to_mri_moving_for_resample.tfm",
        output_dir / "transforms" / case_prefix / "T_mri_moving_to_ct_fixed_for_overlay.tfm",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def manifest_rows_from_metadata_dir(metadata_dir: Path) -> list[dict[str, Any]]:
    return [manifest_row_from_metadata(path) for path in sorted(metadata_dir.glob("*.json"))]


def manifest_row_from_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    paths = payload["paths"]
    grid = payload["grid"]
    registration = payload["manual_registration"]
    counts = registration.get("point_counts", {})
    brightness = payload.get("brightness_trim") or {}
    return {
        "patient_id": str(payload["patient_id"]),
        "ct_series_uid": payload["ct_series_uid"],
        "mri_series_uid": payload["mri_series_uid"],
        "mri_series_description": payload["mri_series_description"],
        "ct_on_mri_hu": paths.get("ct_on_mri_hu", ""),
        "mri_native": paths.get("mri_native", ""),
        "ct_on_mri_norm": paths["ct_on_mri_norm"],
        "mri_norm": paths["mri_norm"],
        "ct_valid_mask_on_mri": paths["ct_valid_mask_on_mri"],
        "metadata_json": str(path),
        "transform_dir": payload["transform_dir"],
        "shape_zyx": json.dumps(grid["shape_zyx"], ensure_ascii=True),
        "pre_brightness_trim_shape_zyx": json.dumps(grid.get("pre_brightness_trim_shape_zyx", ""), ensure_ascii=True),
        "full_shape_zyx": json.dumps(grid["full_shape_zyx"], ensure_ascii=True),
        "spacing_xyz": json.dumps(grid["spacing_xyz"], ensure_ascii=True),
        "origin_lps": json.dumps(grid["origin_lps"], ensure_ascii=True),
        "crop_source": grid["crop_source"],
        "crop_bbox_zyx": json.dumps(grid["crop_bbox_zyx"], ensure_ascii=True),
        "brightness_trim_z_start": brightness.get("z_start_in_pre_trim_volume", ""),
        "brightness_trim_z_end": brightness.get("z_end_in_pre_trim_volume_exclusive", ""),
        "leading_removed": brightness.get("leading_removed", ""),
        "trailing_removed": brightness.get("trailing_removed", ""),
        "total_removed": brightness.get("total_removed", ""),
        "pitch_deg": registration["pitch_deg"],
        "scale_xyz": json.dumps(registration["scale_xyz"], ensure_ascii=True),
        "center_override_count": len(registration.get("center_overrides") or []),
        "ct_l_points": counts.get("CT_L", 0),
        "ct_r_points": counts.get("CT_R", 0),
        "mri_l_points": counts.get("MRI_L", 0),
        "mri_r_points": counts.get("MRI_R", 0),
        "ct_valid_fraction": payload["ct_valid_fraction"],
    }


def build_summary(
    args: argparse.Namespace,
    output_dir: Path,
    all_cases: list[dict[str, Any]],
    exported_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    excluded_patients: set[str],
    brightness_report_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    removed = [int(row["total_removed"]) for row in brightness_report_rows]
    kept_z = []
    for row in brightness_report_rows:
        shape = json.loads(str(row["trimmed_shape_zyx"]))
        kept_z.append(int(shape[0]))
    return {
        "mri_series": args.mri_series,
        "output_grid": "native MRI grid, cropped to CT-valid/MRI-foreground overlap by default",
        "registered_cases_in_db": len(all_cases),
        "exported_cases": len(exported_rows),
        "skipped_cases": len(skipped_rows),
        "excluded_patient_ids": sorted(excluded_patients),
        "output_dir": str(output_dir),
        "excluded_data_issue_patient": "102697",
        "ct_normalization": {"clip_hu": [float(args.ct_clip_low), float(args.ct_clip_high)], "scale": "[-1, 1]"},
        "mri_normalization": {
            "percentiles": [float(args.mri_low_pct), float(args.mri_high_pct)],
            "ignore_zero_voxels": True,
            "scale": "[-1, 1]",
        },
        "brightness_trim": {
            "enabled": not bool(args.no_brightness_trim),
            "method": "patient-wise contiguous z-end low-brightness trim",
            "low_threshold": "max(median - max(mad_multiplier*MAD, min_fraction_drop*median), relative_min_fraction*median)",
            "foreground_threshold_png_like": int(args.foreground_threshold),
            "mad_multiplier": float(args.mad_multiplier),
            "min_fraction_drop": float(args.min_fraction_drop),
            "relative_min_fraction": float(args.relative_min_fraction),
            "min_kept_slices": int(args.min_kept_slices),
            "patients_with_trim": sum(1 for value in removed if value > 0),
            "patients_without_trim": sum(1 for value in removed if value == 0),
            "total_removed_slices": int(sum(removed)),
            "removed_slices_per_case": stats(removed),
            "kept_z_slices_per_case": stats(kept_z),
        },
        "files_per_case": {
            "ct_on_mri_norm": "CT clipped/scaled to [-1, 1]",
            "mri_norm": "MRI percentile-scaled to [-1, 1]",
            "ct_valid_mask_on_mri": "1 where MRI-grid voxel maps inside native CT FOV, else 0",
            "ct_on_mri_hu": "Optional with --save-raw-like: CT HU resampled to native MRI grid",
            "mri_native": "Optional with --save-raw-like: original T2 COR MRI volume on its native grid",
        },
    }


def stats(values: list[int]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "min": float(np.min(arr)) if arr.size else 0.0,
        "median": float(np.median(arr)) if arr.size else 0.0,
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
    }


def create_archive(output_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output_dir, arcname=output_dir.name)


def safe_path_component(value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(value))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "case"


if __name__ == "__main__":
    main()
