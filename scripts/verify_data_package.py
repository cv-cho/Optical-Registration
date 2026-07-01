from __future__ import annotations

import csv
import sys
from pathlib import Path


WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR))

from landmark_workbench.volume import find_series_files, selection_from_queue_row  # noqa: E402


def main() -> None:
    queue_path = WORKDIR / "outputs" / "landmarks" / "work_queue.csv"
    manifest_path = WORKDIR / "reports" / "series_inventory_series.csv"
    if not queue_path.exists():
        raise SystemExit(f"Missing work queue: {queue_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    with queue_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"Work queue is empty: {queue_path}")

    missing_paths: list[str] = []
    file_count_checks: list[str] = []
    for row in rows:
        for modality in ("CT", "MRI"):
            selection = selection_from_queue_row(row, modality)
            if not selection.study_path.exists():
                missing_paths.append(f"{row['patient_id']} {modality}: {selection.study_path}")
                continue
            files = find_series_files(selection)
            if not files:
                file_count_checks.append(f"{row['patient_id']} {modality}: no DICOM files")

    if missing_paths or file_count_checks:
        for item in missing_paths[:20]:
            print(f"missing path: {item}")
        for item in file_count_checks[:20]:
            print(f"missing files: {item}")
        raise SystemExit(2)

    print(f"data package OK: {len(rows)} queued patients")
    print(f"first patient: {rows[0]['patient_id']}")
    print(f"last patient: {rows[-1]['patient_id']}")


if __name__ == "__main__":
    main()
