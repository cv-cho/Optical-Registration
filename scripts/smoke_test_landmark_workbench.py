from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import SimpleITK as sitk


WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR))

from landmark_workbench.geometry import physical_lps_from_napari_zyx  # noqa: E402
from landmark_workbench.qc import left_right_lps_checks, usable_landmark_pairs  # noqa: E402
from landmark_workbench.store import AnnotationStore, LandmarkRecord  # noqa: E402
from landmark_workbench.transform import (  # noqa: E402
    estimate_rigid_initializer,
    save_initialization_result,
    write_resampled_moving_to_fixed,
)


def make_image() -> sitk.Image:
    image = sitk.Image([32, 32, 32], sitk.sitkFloat32)
    image.SetSpacing((1.0, 2.0, 3.0))
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return image


def make_record(label: str, modality: str, point_lps: tuple[float, float, float]) -> LandmarkRecord:
    return LandmarkRecord(
        patient_id="SMOKE",
        study_uid="synthetic",
        series_uid=f"synthetic-{modality}",
        modality=modality,
        landmark_label=label,
        voxel_zyx=[0.0, 0.0, 0.0],
        itk_index_xyz=[0.0, 0.0, 0.0],
        physical_lps_mm=list(point_lps),
        view_used="synthetic",
        slice_index_used=0,
        image_spacing_xyz=[1.0, 1.0, 1.0],
        image_origin_lps=[0.0, 0.0, 0.0],
        image_direction_3x3=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )


def main() -> None:
    output_dir = Path(tempfile.gettempdir()) / "orbit_registration_workbench_smoke"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    db_path = output_dir / "annotations.sqlite"

    image = make_image()
    point_lps = physical_lps_from_napari_zyx(image, (2.0, 3.0, 4.0))
    assert point_lps == (14.0, 26.0, 36.0), point_lps

    fixed_points = {
        "L_GLOBE_MEDIAL_EDGE": (20.0, 0.0, 0.0),
        "L_GLOBE_LATERAL_EDGE": (30.0, 0.0, 0.0),
        "R_GLOBE_MEDIAL_EDGE": (-20.0, 0.0, 0.0),
        "R_GLOBE_LATERAL_EDGE": (-30.0, 0.0, 0.0),
        "L_OPTIC_NERVE_INSERTION": (18.0, 10.0, 0.0),
        "R_OPTIC_NERVE_INSERTION": (-18.0, 10.0, 0.0),
    }
    translation = (5.0, -2.0, 1.0)
    store = AnnotationStore(db_path)
    for label, point in fixed_points.items():
        store.upsert_landmark(make_record(label, "MRI", point))
        moved = tuple(point[i] + translation[i] for i in range(3))
        store.upsert_landmark(make_record(label, "CT", moved))
    alt_record = make_record("L_GLOBE_MEDIAL_EDGE", "MRI", (99.0, 99.0, 99.0))
    alt_record.series_uid = "synthetic-MRI-ALT"
    store.upsert_landmark(alt_record)
    assert len(store.fetch_landmarks("SMOKE", "MRI")) == 7
    assert len(store.fetch_landmarks("SMOKE", "MRI", series_uid="synthetic-MRI")) == 6
    assert len(store.fetch_landmarks("SMOKE", "MRI", series_uid="synthetic-MRI-ALT")) == 1
    store.delete_landmark("SMOKE", "MRI", "L_GLOBE_MEDIAL_EDGE", series_uid="synthetic-MRI-ALT")
    assert len(store.fetch_landmarks("SMOKE", "MRI", series_uid="synthetic-MRI-ALT")) == 0
    records = store.fetch_landmarks("SMOKE")
    assert usable_landmark_pairs(records) == [
        "L_GLOBE_DERIVED_CENTER",
        "L_OPTIC_NERVE_INSERTION",
        "R_GLOBE_DERIVED_CENTER",
        "R_OPTIC_NERVE_INSERTION",
    ]
    assert all(item["pass"] for item in left_right_lps_checks(records))
    result = estimate_rigid_initializer(records, fixed_modality="MRI", moving_modality="CT")
    assert result.status == "pass", result
    assert result.max_residual_mm < 1e-5, result.max_residual_mm
    save_initialization_result(result, output_dir / "initialization")
    write_resampled_moving_to_fixed(
        moving_image=image,
        fixed_image=image,
        transform_fixed_to_moving=result.transform_fixed_to_moving,
        output_path=output_dir / "initialization" / "synthetic_resampled.nii.gz",
        default_value=0,
    )
    store.export_jsonl(output_dir / "annotations.jsonl")
    store.close()
    print(f"smoke test passed: {output_dir}")


if __name__ == "__main__":
    main()
