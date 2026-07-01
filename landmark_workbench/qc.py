from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .schema import LANDMARK_BY_LABEL, paired_left_right_labels


GLOBE_EDGE_LABELS = {
    "L_GLOBE_MEDIAL_EDGE",
    "L_GLOBE_LATERAL_EDGE",
    "R_GLOBE_MEDIAL_EDGE",
    "R_GLOBE_LATERAL_EDGE",
}

GLOBE_EDGE_PAIRS = {
    "L": ("L_GLOBE_MEDIAL_EDGE", "L_GLOBE_LATERAL_EDGE", "L_GLOBE_DERIVED_CENTER"),
    "R": ("R_GLOBE_MEDIAL_EDGE", "R_GLOBE_LATERAL_EDGE", "R_GLOBE_DERIVED_CENTER"),
}


def registration_landmark_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable_records = [record for record in records if _record_is_transform_usable(record)]
    transformed: list[dict[str, Any]] = [
        dict(record)
        for record in usable_records
        if str(record.get("landmark_label")) not in GLOBE_EDGE_LABELS
    ]
    by_key = {
        (str(record["modality"]).upper(), str(record["landmark_label"])): record
        for record in usable_records
        if record.get("physical_lps_mm") is not None
    }
    for modality in ("CT", "MRI"):
        for _side, (medial_label, lateral_label, derived_label) in GLOBE_EDGE_PAIRS.items():
            medial = by_key.get((modality, medial_label))
            lateral = by_key.get((modality, lateral_label))
            if medial is None or lateral is None:
                continue
            medial_point = np.asarray(medial["physical_lps_mm"], dtype=float)
            lateral_point = np.asarray(lateral["physical_lps_mm"], dtype=float)
            derived = dict(medial)
            derived["landmark_label"] = derived_label
            derived["physical_lps_mm"] = ((medial_point + lateral_point) * 0.5).tolist()
            derived["source"] = "derived_from_globe_edges"
            derived["visibility"] = "uncertain" if "uncertain" in {medial.get("visibility"), lateral.get("visibility")} else "visible"
            derived["quality"] = max(int(medial.get("quality", 0)), int(lateral.get("quality", 0)))
            derived["use_for_transform"] = True
            transformed.append(derived)
    return transformed


def usable_landmark_pairs(records: list[dict[str, Any]]) -> list[str]:
    by_modality: dict[str, set[str]] = defaultdict(set)
    for record in registration_landmark_records(records):
        by_modality[str(record["modality"]).upper()].add(str(record["landmark_label"]))
    return sorted(by_modality["CT"].intersection(by_modality["MRI"]))


def _record_is_transform_usable(record: dict[str, Any]) -> bool:
    if not record.get("use_for_transform", True):
        return False
    if record.get("visibility") not in ("visible", "uncertain"):
        return False
    return True


def left_right_lps_checks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (str(record["modality"]).upper(), str(record["landmark_label"])): record
        for record in records
        if record.get("physical_lps_mm") is not None
    }
    checks: list[dict[str, Any]] = []
    for modality in ("CT", "MRI"):
        for left_label, right_label in paired_left_right_labels():
            left = by_key.get((modality, left_label))
            right = by_key.get((modality, right_label))
            if not left or not right:
                continue
            left_x = float(left["physical_lps_mm"][0])
            right_x = float(right["physical_lps_mm"][0])
            checks.append(
                {
                    "modality": modality,
                    "left_label": left_label,
                    "right_label": right_label,
                    "left_x_lps": left_x,
                    "right_x_lps": right_x,
                    "pass": left_x > right_x,
                    "message": "LPS x should increase toward patient left.",
                }
            )
    return checks


def landmark_shape_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"distances_mm": {}}
    by_label = {str(record["landmark_label"]): record for record in records}
    for left_label, right_label in paired_left_right_labels():
        if left_label not in by_label or right_label not in by_label:
            continue
        left = np.asarray(by_label[left_label]["physical_lps_mm"], dtype=float)
        right = np.asarray(by_label[right_label]["physical_lps_mm"], dtype=float)
        structure = LANDMARK_BY_LABEL[left_label].structure
        summary["distances_mm"][f"{structure}_left_right"] = float(np.linalg.norm(left - right))
    return summary
