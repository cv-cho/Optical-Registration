from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from pathlib import Path


WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR))


def configure_qt_runtime() -> None:
    os.environ["QT_API"] = "pyqt6"
    plugin_root = pyqt6_plugin_root()
    if plugin_root is not None:
        platforms = plugin_root / "platforms"
        if platforms.exists():
            if not os.environ.get("QT_PLUGIN_PATH"):
                os.environ["QT_PLUGIN_PATH"] = str(plugin_root)
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms)
        qt_bin = plugin_root.parent / "bin"
        if qt_bin.exists():
            os.environ["PATH"] = str(qt_bin) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(qt_bin))
    if sys.platform == "darwin":
        os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")


def pyqt6_plugin_root() -> Path | None:
    try:
        from PyQt6.QtCore import QLibraryInfo

        path = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
        if path.exists():
            return path
    except Exception:
        pass
    candidates = [
        WORKDIR / ".venv" / "Lib" / "site-packages" / "PyQt6" / "Qt6" / "plugins",
        WORKDIR.parent / ".venv-rtx5090" / "Lib" / "site-packages" / "PyQt6" / "Qt6" / "plugins",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


configure_qt_runtime()

from landmark_workbench.dual_view_app import run  # noqa: E402


DEFAULT_PATIENT_ID = "101195"
DEFAULT_MRI_SERIES = "T2 COR dixon_(IN W)_in"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the dual-panel CT/MRI landmark workbench.")
    parser.add_argument("--patient-id", default="", help="Patient to open. Omit to resume from the last edited patient.")
    parser.add_argument("--manifest", default=str(WORKDIR / "reports" / "series_inventory_series.csv"))
    parser.add_argument("--db", default=str(WORKDIR / "outputs" / "landmarks" / "annotations.sqlite"))
    parser.add_argument("--work-queue", default=str(WORKDIR / "outputs" / "landmarks" / "work_queue.csv"))
    parser.add_argument("--ct-series", default="AX")
    parser.add_argument("--mri-series", default=DEFAULT_MRI_SERIES)
    parser.add_argument("--annotator-id", default="default")
    args = parser.parse_args()
    patient_id = resolve_start_patient(
        requested_patient_id=args.patient_id,
        db_path=Path(args.db),
        manifest_csv=Path(args.manifest),
        work_queue_csv=Path(args.work_queue),
        mri_series_description=args.mri_series,
    )
    run(
        manifest_csv=args.manifest,
        patient_id=patient_id,
        db_path=args.db,
        ct_series_description=args.ct_series,
        mri_series_description=args.mri_series,
        work_queue_csv=args.work_queue,
        annotator_id=args.annotator_id,
    )


def resolve_start_patient(
    requested_patient_id: str,
    db_path: Path,
    manifest_csv: Path,
    work_queue_csv: Path,
    mri_series_description: str,
) -> str:
    requested = str(requested_patient_id or "").strip()
    if requested:
        return requested
    latest = latest_globe_patient(db_path, manifest_csv, mri_series_description)
    if latest:
        print(f"No --patient-id supplied; resuming from last edited patient {latest}.")
        return latest
    fallback = first_queued_patient_with_mri(work_queue_csv, manifest_csv, mri_series_description)
    if fallback:
        print(f"No saved globe points found; starting from first queued {mri_series_description} patient {fallback}.")
        return fallback
    print(f"No saved globe points found; starting from fallback patient {DEFAULT_PATIENT_ID}.")
    return DEFAULT_PATIENT_ID


def latest_globe_patient(db_path: Path, manifest_csv: Path, mri_series_description: str) -> str:
    if not db_path.exists():
        return ""
    eligible = patients_with_mri_series(manifest_csv, mri_series_description)
    target_mri_uids = mri_series_uids(manifest_csv, mri_series_description)
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT patient_id, updated_at
            FROM globe_manual_parameters
            WHERE mri_series_description=?
            ORDER BY updated_at DESC, patient_id DESC
            """,
            (str(mri_series_description),),
        ).fetchall()
        for patient_id, _latest in rows:
            patient_id = str(patient_id)
            if not eligible or patient_id in eligible:
                return patient_id

        rows = con.execute(
            """
            SELECT patient_id, series_uid, created_at
            FROM globe_surface_points
            WHERE modality='MRI'
            ORDER BY created_at DESC, patient_id DESC
            """
        ).fetchall()
    except sqlite3.Error:
        return ""
    finally:
        con.close()
    for patient_id, series_uid, _latest in rows:
        patient_id = str(patient_id)
        if target_mri_uids and str(series_uid) not in target_mri_uids:
            continue
        if not eligible or patient_id in eligible:
            return patient_id
    return ""


def first_queued_patient_with_mri(work_queue_csv: Path, manifest_csv: Path, mri_series_description: str) -> str:
    eligible = patients_with_mri_series(manifest_csv, mri_series_description)
    if not work_queue_csv.exists():
        return next(iter(sorted(eligible)), "") if eligible else ""
    with work_queue_csv.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            patient_id = str(row.get("patient_id", "")).strip()
            if patient_id and (not eligible or patient_id in eligible):
                return patient_id
    return ""


def patients_with_mri_series(manifest_csv: Path, mri_series_description: str) -> set[str]:
    if not manifest_csv.exists():
        return set()
    patients: set[str] = set()
    with manifest_csv.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if str(row.get("modality", "")).upper() != "MR":
                continue
            if str(row.get("series_description", "")) != str(mri_series_description):
                continue
            patient_id = str(row.get("patient_id", "")).strip()
            if patient_id:
                patients.add(patient_id)
    return patients


def mri_series_uids(manifest_csv: Path, mri_series_description: str) -> set[str]:
    if not manifest_csv.exists():
        return set()
    series_uids: set[str] = set()
    with manifest_csv.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if str(row.get("modality", "")).upper() != "MR":
                continue
            if str(row.get("series_description", "")) != str(mri_series_description):
                continue
            series_uid = str(row.get("series_uid", "")).strip()
            if series_uid:
                series_uids.add(series_uid)
    return series_uids


if __name__ == "__main__":
    main()
