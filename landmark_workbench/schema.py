from __future__ import annotations

from dataclasses import dataclass


SIDES = ("L", "R")
MODALITIES = ("CT", "MRI")
VISIBILITY_VALUES = ("visible", "uncertain", "not_visible", "outside_fov")
SOURCE_VALUES = ("manual", "auto", "auto_corrected")
QC_STATUS_VALUES = ("unchecked", "pass", "warning", "fail")


@dataclass(frozen=True)
class LandmarkSpec:
    label: str
    side: str
    structure: str
    required: bool
    transform_priority: int
    description: str


LANDMARKS: tuple[LandmarkSpec, ...] = (
    LandmarkSpec(
        "L_GLOBE_MEDIAL_EDGE",
        "L",
        "globe_medial_edge",
        True,
        1,
        "Medial/nasal edge point of the left globe contour.",
    ),
    LandmarkSpec(
        "L_GLOBE_LATERAL_EDGE",
        "L",
        "globe_lateral_edge",
        True,
        1,
        "Lateral/temporal edge point of the left globe contour.",
    ),
    LandmarkSpec(
        "R_GLOBE_MEDIAL_EDGE",
        "R",
        "globe_medial_edge",
        True,
        1,
        "Medial/nasal edge point of the right globe contour.",
    ),
    LandmarkSpec(
        "R_GLOBE_LATERAL_EDGE",
        "R",
        "globe_lateral_edge",
        True,
        1,
        "Lateral/temporal edge point of the right globe contour.",
    ),
    LandmarkSpec(
        "L_OPTIC_NERVE_INSERTION",
        "L",
        "optic_nerve_insertion",
        False,
        2,
        "Posterior globe pole where the left optic nerve inserts.",
    ),
    LandmarkSpec(
        "R_OPTIC_NERVE_INSERTION",
        "R",
        "optic_nerve_insertion",
        False,
        2,
        "Posterior globe pole where the right optic nerve inserts.",
    ),
    LandmarkSpec(
        "L_OPTIC_CANAL_ENTRANCE",
        "L",
        "optic_canal_entrance",
        False,
        2,
        "Center of the left anterior bony optic canal opening.",
    ),
    LandmarkSpec(
        "R_OPTIC_CANAL_ENTRANCE",
        "R",
        "optic_canal_entrance",
        False,
        2,
        "Center of the right anterior bony optic canal opening.",
    ),
    LandmarkSpec(
        "L_AUXILIARY_INDICATOR",
        "L",
        "auxiliary_indicator",
        False,
        3,
        "Left-side auxiliary point for additional registration guidance.",
    ),
    LandmarkSpec(
        "R_AUXILIARY_INDICATOR",
        "R",
        "auxiliary_indicator",
        False,
        3,
        "Right-side auxiliary point for additional registration guidance.",
    ),
)

LANDMARK_BY_LABEL = {item.label: item for item in LANDMARKS}
LANDMARK_LABELS = tuple(item.label for item in LANDMARKS)
LANDMARK_SHORTCUTS = {
    "q": "L_GLOBE_MEDIAL_EDGE",
    "w": "L_GLOBE_LATERAL_EDGE",
    "e": "R_GLOBE_MEDIAL_EDGE",
    "r": "R_GLOBE_LATERAL_EDGE",
    "a": "L_OPTIC_NERVE_INSERTION",
    "s": "R_OPTIC_NERVE_INSERTION",
    "d": "L_OPTIC_CANAL_ENTRANCE",
    "f": "R_OPTIC_CANAL_ENTRANCE",
    "g": "L_AUXILIARY_INDICATOR",
    "h": "R_AUXILIARY_INDICATOR",
}


def validate_landmark_label(label: str) -> str:
    if label not in LANDMARK_BY_LABEL:
        raise ValueError(f"Unknown landmark label: {label}")
    return label


def validate_modality(modality: str) -> str:
    value = modality.upper()
    if value not in MODALITIES:
        raise ValueError(f"Unknown modality: {modality}")
    return value


def opposite_label(label: str) -> str | None:
    spec = LANDMARK_BY_LABEL.get(label)
    if spec is None:
        return None
    if spec.side == "L":
        return "R_" + label[2:]
    if spec.side == "R":
        return "L_" + label[2:]
    return None


def paired_left_right_labels() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for spec in LANDMARKS:
        if spec.side != "L":
            continue
        right = opposite_label(spec.label)
        if right in LANDMARK_BY_LABEL:
            pairs.append((spec.label, right))
    return pairs
