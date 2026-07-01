from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR))

from landmark_workbench.store import AnnotationStore  # noqa: E402
from landmark_workbench.transform import (  # noqa: E402
    estimate_rigid_initializer,
    result_summary,
    save_initialization_result,
    write_resampled_moving_to_fixed,
)
from landmark_workbench.volume import (  # noqa: E402
    load_patient_pair,
    load_patient_pair_by_uid,
    load_series_volume,
    select_series,
    selection_from_queue_row,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute landmark-based CT-to-MRI initialization.")
    parser.add_argument("--patient-id", default="101195")
    parser.add_argument("--manifest", default=str(WORKDIR / "reports" / "series_inventory_series.csv"))
    parser.add_argument("--db", default=str(WORKDIR / "outputs" / "landmarks" / "annotations.sqlite"))
    parser.add_argument("--work-queue", default=str(WORKDIR / "outputs" / "landmarks" / "work_queue.csv"))
    parser.add_argument("--ct-series", default="AX")
    parser.add_argument("--mri-series", default="T2 COR dixon_(IN W)_in")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Annotation DB does not exist: {db_path}")

    queue_row = read_queue_row(Path(args.work_queue), args.patient_id)
    if (
        queue_row
        and queue_row.get("ct_series_uid")
        and queue_row.get("mri_series_uid")
        and queue_row.get("mri_series_description") == args.mri_series
    ):
        ct, mri = load_patient_pair_by_uid(
            args.manifest,
            args.patient_id,
            queue_row["ct_series_uid"],
            queue_row["mri_series_uid"],
            queue_row=queue_row,
        )
    elif queue_row and queue_row.get("ct_series_uid"):
        ct = load_series_volume(selection_from_queue_row(queue_row, "CT"))
        mri = load_series_volume(select_series(args.manifest, args.patient_id, "MR", args.mri_series))
    else:
        ct, mri = load_patient_pair(
            args.manifest,
            args.patient_id,
            ct_series_description=args.ct_series,
            mri_series_description=args.mri_series,
        )
    store = AnnotationStore(db_path)
    records = []
    records.extend(store.fetch_landmarks(args.patient_id, "CT", series_uid=ct.selection.series_uid))
    records.extend(store.fetch_landmarks(args.patient_id, "MRI", series_uid=mri.selection.series_uid))
    store.close()

    try:
        result = estimate_rigid_initializer(records, fixed_modality="MRI", moving_modality="CT")
    except Exception as exc:
        print(f"Initialization failed: {exc}")
        labels_by_modality = {
            modality: sorted(record["landmark_label"] for record in records if record["modality"] == modality)
            for modality in ("CT", "MRI")
        }
        print(json.dumps({"patient_id": args.patient_id, "labels_by_modality": labels_by_modality}, indent=2))
        raise SystemExit(2)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else db_path.parent / args.patient_id / "initialization" / safe_path_component(mri.selection.series_description)
    )
    save_initialization_result(result, output_dir)
    write_resampled_moving_to_fixed(
        moving_image=ct.image,
        fixed_image=mri.image,
        transform_fixed_to_moving=result.transform_fixed_to_moving,
        output_path=output_dir / "ct_resampled_to_mri_init.nii.gz",
        default_value=-1024.0,
    )
    summary = result_summary(result)
    summary["output_dir"] = str(output_dir)
    print(json.dumps(summary, indent=2))


def read_queue_row(path: Path, patient_id: str) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("patient_id") == str(patient_id):
                return dict(row)
    return None


def safe_path_component(value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in value)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "series"


if __name__ == "__main__":
    main()
