from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path


WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR))

from landmark_workbench.qc import left_right_lps_checks, usable_landmark_pairs  # noqa: E402


def load_rows(db_path: Path, patient_id: str) -> list[dict[str, object]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT *
        FROM landmarks
        WHERE patient_id = ?
        ORDER BY modality, landmark_label
        """,
        (patient_id,),
    ).fetchall()
    conn.close()
    return [deserialize(dict(row)) for row in rows]


def deserialize(row: dict[str, object]) -> dict[str, object]:
    for key in (
        "voxel_zyx",
        "itk_index_xyz",
        "physical_lps_mm",
        "image_spacing_xyz",
        "image_origin_lps",
        "image_direction_3x3",
        "software_versions",
    ):
        row[key] = json.loads(str(row[key]))
    row["use_for_transform"] = bool(row["use_for_transform"])
    return row


def event_summary(db_path: Path, patient_id: str) -> Counter[tuple[str, str]]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT modality, landmark_label, COUNT(*) AS n
        FROM annotation_events
        WHERE patient_id = ?
        GROUP BY modality, landmark_label
        ORDER BY modality, landmark_label
        """,
        (patient_id,),
    ).fetchall()
    conn.close()
    return Counter({(str(row[0]), str(row[1])): int(row[2]) for row in rows})


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect saved CT/MRI landmark annotations.")
    parser.add_argument("--db", default=str(WORKDIR / "outputs" / "landmarks" / "annotations.sqlite"))
    parser.add_argument("--patient-id", default="101195")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Annotation DB does not exist: {db_path}")

    rows = load_rows(db_path, args.patient_id)
    events = event_summary(db_path, args.patient_id)
    print(f"DB: {db_path}")
    print(f"patient_id: {args.patient_id}")
    print(f"final landmark rows: {len(rows)}")
    print(f"annotation events: {sum(events.values())}")
    if events:
        print("event counts by modality/label:")
        for (modality, label), count in sorted(events.items()):
            print(f"  {modality:3s} {label:28s} {count}")

    by_modality: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_modality[str(row["modality"])].append(row)
    for modality in ("CT", "MRI"):
        print(f"\n{modality} final landmarks: {len(by_modality[modality])}")
        for row in by_modality[modality]:
            print(
                "  {label:28s} voxel_zyx={voxel} itk_xyz={itk} lps_mm={lps} visibility={visibility} quality={quality}".format(
                    label=str(row["landmark_label"]),
                    voxel=_round_list(row["voxel_zyx"]),
                    itk=_round_list(row["itk_index_xyz"]),
                    lps=_round_list(row["physical_lps_mm"]),
                    visibility=row["visibility"],
                    quality=row["quality"],
                )
            )

    usable = usable_landmark_pairs(rows)
    print(f"\nusable CT/MRI pairs: {len(usable)} {usable}")
    checks = left_right_lps_checks(rows)
    if checks:
        print("left/right LPS checks:")
        for item in checks:
            status = "PASS" if item["pass"] else "FAIL"
            print(
                f"  {status} {item['modality']} {item['left_label']} x={item['left_x_lps']:.2f} "
                f"> {item['right_label']} x={item['right_x_lps']:.2f}"
            )
    else:
        print("left/right LPS checks: not enough paired L/R labels yet")


def _round_list(values: object) -> list[float]:
    return [round(float(v), 3) for v in values]  # type: ignore[arg-type]


if __name__ == "__main__":
    main()

