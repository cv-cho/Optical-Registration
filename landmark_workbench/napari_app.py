from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import (
    napari_zyx_from_physical_lps,
    napari_zyx_to_itk_xyz,
    physical_lps_from_napari_zyx,
    point_inside_array_shape_zyx,
)
from .qc import left_right_lps_checks, usable_landmark_pairs
from .schema import LANDMARK_LABELS, LANDMARK_SHORTCUTS
from .store import AnnotationStore, LandmarkRecord
from .transform import (
    estimate_rigid_initializer,
    result_summary,
    save_initialization_result,
    write_resampled_moving_to_fixed,
)
from .volume import (
    ct_window,
    load_series_volume,
    normalize_percentile,
    read_manifest_rows,
    select_series,
    selection_from_manifest_row,
    selection_from_queue_row,
)


PATIENT_LAYER_NAMES = {
    "CT_soft",
    "CT_bone",
    "MRI_T2",
    "MRI_selected",
    "CT_landmarks",
    "MRI_landmarks",
    "CT_soft_on_MRI_init",
    "CT_bone_on_MRI_init",
    "CT_landmarks_projected_to_MRI_init",
    "MRI_selected_axial_recon",
    "CT_soft_on_MRI_axial_init",
    "CT_bone_on_MRI_axial_init",
    "CT_landmarks_projected_to_MRI_axial_init",
}

MRI_CANDIDATE_SERIES = (
    "T2 COR dixon_(IN W)_in",
    "T2 COR dixon_(IN W)_F",
    "t2_tse_dixon_tra_384_2mm_in",
    "t2_tse_dixon_tra_384_2mm_W",
    "AXL T1 MPRAGE FS POST",
)

MRI_CANDIDATE_SHORT_NAMES = {
    "T2 COR dixon_(IN W)_in": "T2 COR in",
    "T2 COR dixon_(IN W)_F": "T2 COR fat",
    "t2_tse_dixon_tra_384_2mm_in": "T2 AX in",
    "t2_tse_dixon_tra_384_2mm_W": "T2 AX water",
    "AXL T1 MPRAGE FS POST": "T1 AX post",
}

class PointCompletionPanel:
    def __init__(self, workbench: "LandmarkWorkbench"):
        from qtpy.QtWidgets import QGridLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

        self.workbench = workbench
        self.widget = QWidget()
        root = QVBoxLayout(self.widget)
        self.patient_label = QLabel()
        self.patient_label.setWordWrap(True)
        root.addWidget(self.patient_label)
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}

        for modality in ("CT", "MRI"):
            group = QGroupBox(f"{modality} Points")
            grid = QGridLayout(group)
            grid.setColumnStretch(1, 1)
            grid.addWidget(QLabel("Label"), 0, 0)
            grid.addWidget(QLabel("State"), 0, 1)
            grid.addWidget(QLabel("Del"), 0, 2)
            for row_index, label in enumerate(LANDMARK_LABELS, start=1):
                label_widget = QLabel(label)
                label_widget.setToolTip(label)
                status_widget = QLabel("needs work")
                status_widget.setWordWrap(True)
                delete_button = QPushButton("X")
                delete_button.setFixedWidth(30)
                delete_button.setEnabled(False)
                delete_button.setToolTip(f"Delete saved {modality} {label}")
                delete_button.clicked.connect(
                    lambda checked=False, m=modality, l=label: self.workbench.delete_saved_landmark(m, l)
                )
                grid.addWidget(label_widget, row_index, 0)
                grid.addWidget(status_widget, row_index, 1)
                grid.addWidget(delete_button, row_index, 2)
                self.rows[(modality, label)] = {
                    "label": label_widget,
                    "status": status_widget,
                    "button": delete_button,
                }
            root.addWidget(group)

        root.addStretch(1)
        self.refresh()

    def refresh(self) -> None:
        records = self.workbench.current_landmark_records()
        by_key = {(record["modality"], record["landmark_label"]): record for record in records}
        self.patient_label.setText(
            f"Patient {self.workbench.patient_id} | queue {self.workbench._queue_position_text()}\n"
            f"MRI: {MRI_CANDIDATE_SHORT_NAMES.get(self.workbench.mri_series_description, self.workbench.mri_series_description)}"
        )
        for modality in ("CT", "MRI"):
            for label in LANDMARK_LABELS:
                widgets = self.rows[(modality, label)]
                record = by_key.get((modality, label))
                is_current = modality == self.workbench.current_modality and label == self.workbench.current_label
                widgets["label"].setStyleSheet("font-weight: bold;" if is_current else "")
                if record:
                    voxel_zyx = record.get("voxel_zyx") or [0.0, 0.0, 0.0]
                    status = f"done {record['visibility']} q{record['quality']} z={float(voxel_zyx[0]):.1f}"
                    widgets["status"].setText(status)
                    widgets["status"].setStyleSheet("color: #1b7f3a;")
                    widgets["button"].setEnabled(True)
                else:
                    widgets["status"].setText("needs work")
                    widgets["status"].setStyleSheet("color: #777777;")
                    widgets["button"].setEnabled(False)


class LandmarkWorkbench:
    def __init__(
        self,
        manifest_csv: str | Path,
        patient_id: str,
        db_path: str | Path,
        ct_series_description: str = "AX",
        mri_series_description: str = "T2 COR dixon_(IN W)_in",
        work_queue_csv: str | Path | None = None,
        window_scale: float = 2.0,
        annotator_id: str = "default",
    ):
        self.manifest_csv = Path(manifest_csv)
        self.manifest_rows = read_manifest_rows(self.manifest_csv)
        self.work_queue_csv = Path(work_queue_csv) if work_queue_csv else None
        self.queue_rows = self._read_work_queue(self.work_queue_csv)
        self._mri_candidate_cache: dict[str, dict[str, dict[str, str]]] = {}
        self.patient_id = str(patient_id)
        self.queue_index = self._queue_index_for_patient(self.patient_id)
        self.db_path = Path(db_path)
        self.ct_series_description = ct_series_description
        self.mri_series_description = mri_series_description
        self.requested_mri_series_description = mri_series_description
        self.output_dir = self._initialization_output_dir(self.patient_id, self.mri_series_description)
        self.window_scale = float(window_scale)
        self.annotator_id = annotator_id
        self.store = AnnotationStore(self.db_path)
        self.current_label = LANDMARK_LABELS[0]
        self.current_modality = "CT"
        self.current_visibility = "visible"
        self.current_quality = 0
        self._updating_layers = False
        self.annotation_state_widget = None
        self.mri_series_widget = None
        self.point_completion_panel = None
        self.viewer = None
        self.ct_volume = None
        self.mri_volume = None
        self.ct_points = None
        self.mri_points = None

    def launch(self) -> None:
        try:
            import napari
            from magicgui import magicgui
        except ImportError as exc:
            raise RuntimeError(
                "napari and magicgui are required for the GUI. "
                "Install requirements-landmark-workbench.txt first."
            ) from exc

        viewer = napari.Viewer(title=f"Landmark workbench - {self.patient_id}")
        self.viewer = viewer
        self._resize_viewer_window()

        @magicgui(
            auto_call=True,
            target_modality={"choices": ["CT", "MRI"]},
            label={"choices": LANDMARK_LABELS},
            visibility={"choices": ["visible", "uncertain", "not_visible", "outside_fov"]},
            quality={"min": 0, "max": 2},
        )
        def annotation_state(
            target_modality: str = "CT",
            label: str = LANDMARK_LABELS[0],
            visibility: str = "visible",
            quality: int = 0,
        ) -> None:
            previous_modality = self.current_modality
            self.current_modality = target_modality
            self.current_label = label
            self.current_visibility = visibility
            self.current_quality = int(quality)
            if target_modality != previous_modality:
                self._show_modality_for_annotation(target_modality)
            self._activate_points_layer(target_modality)
        self.annotation_state_widget = annotation_state

        @magicgui(call_button="Load patient", result_widget=True)
        def load_patient(patient_id: str = self.patient_id) -> str:
            return self.load_patient(str(patient_id))

        @magicgui(call_button="Previous patient", result_widget=True)
        def previous_patient() -> str:
            return self.previous_patient()

        @magicgui(call_button="Next patient", result_widget=True)
        def next_patient() -> str:
            return self.next_patient()

        @magicgui(call_button="Refresh status", result_widget=True)
        def refresh_status() -> str:
            return self.status_text()

        @magicgui(call_button="Delete current label", result_widget=True)
        def delete_current_label() -> str:
            return self.delete_saved_landmark(self.current_modality, self.current_label)

        @magicgui(
            call_button="Load MRI series",
            result_widget=True,
            mri_series={"choices": MRI_CANDIDATE_SERIES},
        )
        def mri_series_selector(mri_series: str = self.requested_mri_series_description) -> str:
            return self.load_mri_series(str(mri_series))
        self.mri_series_widget = mri_series_selector

        @magicgui(call_button="MRI availability", result_widget=True)
        def mri_availability() -> str:
            return self.mri_availability_text()

        @magicgui(call_button="Previous patient with selected MRI", result_widget=True)
        def previous_patient_with_mri() -> str:
            return self.previous_patient_with_current_mri_series()

        @magicgui(call_button="Next patient with selected MRI", result_widget=True)
        def next_patient_with_mri() -> str:
            return self.next_patient_with_current_mri_series()

        @magicgui(call_button="Compute init + overlay", result_widget=True)
        def compute_initialization() -> str:
            return self.compute_initialization_overlay()

        self.point_completion_panel = PointCompletionPanel(self)

        viewer.window.add_dock_widget(annotation_state, area="right", name="Annotation state")
        viewer.window.add_dock_widget(
            self.point_completion_panel.widget,
            area="right",
            name="Current patient points",
        )
        viewer.window.add_dock_widget(load_patient, area="right", name="Patient loader")
        viewer.window.add_dock_widget(previous_patient, area="right", name="Previous patient")
        viewer.window.add_dock_widget(next_patient, area="right", name="Next patient")
        viewer.window.add_dock_widget(refresh_status, area="right", name="Saved point status")
        viewer.window.add_dock_widget(delete_current_label, area="right", name="Point edit")
        viewer.window.add_dock_widget(mri_series_selector, area="right", name="MRI series selector")
        viewer.window.add_dock_widget(mri_availability, area="right", name="MRI candidate status")
        viewer.window.add_dock_widget(previous_patient_with_mri, area="right", name="Previous candidate patient")
        viewer.window.add_dock_widget(next_patient_with_mri, area="right", name="Next candidate patient")
        viewer.window.add_dock_widget(compute_initialization, area="right", name="Initialization QC")
        self._bind_keys()
        self.load_patient(self.patient_id)
        napari.run()

    def _on_points_changed(self, modality: str) -> None:
        if self._updating_layers:
            return
        layer = self.ct_points if modality == "CT" else self.mri_points
        volume = self.ct_volume if modality == "CT" else self.mri_volume
        if layer is None or volume is None:
            return
        data = np.asarray(layer.data, dtype=float)
        if data.size == 0:
            return
        point_zyx = data[-1].tolist()
        if not point_inside_array_shape_zyx(point_zyx, volume.array_zyx.shape):
            self._restore_saved_points_after_rejected_click()
            shape_text = "x".join(str(v) for v in volume.array_zyx.shape)
            self._notify_status(
                f"Rejected {modality} point outside active volume shape zyx={shape_text}: {point_zyx}"
            )
            return
        index_xyz = list(napari_zyx_to_itk_xyz(point_zyx))
        physical_lps = list(physical_lps_from_napari_zyx(volume.image, point_zyx))
        record = LandmarkRecord(
            patient_id=self.patient_id,
            study_uid=volume.selection.study_uid,
            series_uid=volume.selection.series_uid,
            modality=modality,
            landmark_label=self.current_label,
            voxel_zyx=[float(v) for v in point_zyx],
            itk_index_xyz=[float(v) for v in index_xyz],
            physical_lps_mm=[float(v) for v in physical_lps],
            view_used="napari_native_voxel",
            slice_index_used=float(point_zyx[0]),
            image_spacing_xyz=[float(v) for v in volume.image.GetSpacing()],
            image_origin_lps=[float(v) for v in volume.image.GetOrigin()],
            image_direction_3x3=volume.geometry["direction_3x3"],
            source="manual",
            visibility=self.current_visibility,
            use_for_transform=self.current_visibility in ("visible", "uncertain"),
            quality=self.current_quality,
            annotator_id=self.annotator_id,
        )
        self.store.upsert_landmark(record)
        self.refresh_points()

    def refresh_points(self) -> None:
        self._updating_layers = True
        try:
            self._refresh_layer("CT", self.ct_points)
            self._refresh_layer("MRI", self.mri_points)
        finally:
            self._updating_layers = False
        self._refresh_point_completion_panel()

    def delete_saved_landmark(self, modality: str, label: str) -> str:
        series_uid = self._series_uid_for_modality(modality)
        self.store.delete_landmark(self.patient_id, modality, label, series_uid=series_uid)
        self.refresh_points()
        message = f"Deleted {modality} {label}"
        self._notify_status(message)
        return message

    def _refresh_layer(self, modality: str, layer: Any) -> None:
        if layer is None:
            return
        records = self.store.fetch_landmarks(
            self.patient_id,
            modality,
            series_uid=self._series_uid_for_modality(modality),
        )
        if records:
            layer.data = np.asarray([record["voxel_zyx"] for record in records], dtype=float)
            layer.features = {
                "label": [record["landmark_label"] for record in records],
                "visibility": [record["visibility"] for record in records],
                "source": [record["source"] for record in records],
                "quality": [record["quality"] for record in records],
            }
        else:
            layer.data = np.empty((0, 3))
            layer.features = {"label": [], "visibility": [], "source": [], "quality": []}

    def current_landmark_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        ct_uid = self._series_uid_for_modality("CT")
        mri_uid = self._series_uid_for_modality("MRI")
        if ct_uid:
            records.extend(self.store.fetch_landmarks(self.patient_id, "CT", series_uid=ct_uid))
        if mri_uid:
            records.extend(self.store.fetch_landmarks(self.patient_id, "MRI", series_uid=mri_uid))
        return records

    def _series_uid_for_modality(self, modality: str) -> str | None:
        modality = modality.upper()
        if modality == "CT" and self.ct_volume is not None:
            return self.ct_volume.selection.series_uid
        if modality == "MRI" and self.mri_volume is not None:
            return self.mri_volume.selection.series_uid
        return None

    def load_patient(self, patient_id: str) -> str:
        if self.viewer is None:
            return "Viewer is not ready."
        patient_id = str(patient_id).strip()
        if not patient_id:
            return "Patient id is empty."
        self._updating_layers = True
        try:
            queue_row = self._queue_row_for_patient(patient_id)
            ct_volume = self._load_ct_volume(patient_id, queue_row)
            mri_volume = self._load_mri_volume(
                patient_id,
                self.requested_mri_series_description,
                queue_row,
            )
            self.ct_volume = ct_volume
            self.mri_volume = mri_volume
            self.patient_id = patient_id
            self.queue_index = self._queue_index_for_patient(patient_id)
            self.ct_series_description = self.ct_volume.selection.series_description
            self.mri_series_description = self.mri_volume.selection.series_description
            self.requested_mri_series_description = self.mri_series_description
            self.output_dir = self._initialization_output_dir(self.patient_id, self.mri_series_description)
            self._replace_patient_layers()
            self.viewer.title = f"Landmark workbench - {self.patient_id}"
            self._set_mri_series_widget_value(self.mri_series_description)
            self._show_modality_for_annotation(self.current_modality)
            self._activate_points_layer(self.current_modality)
        except Exception as exc:
            return f"Failed to load patient {patient_id}: {exc}"
        finally:
            self._updating_layers = False
        self.refresh_points()
        return self.status_text()

    def next_patient(self) -> str:
        return self._neighbor_patient_with_mri_series(forward=True)

    def previous_patient(self) -> str:
        return self._neighbor_patient_with_mri_series(forward=False)

    def load_mri_series(self, series_description: str) -> str:
        series_description = str(series_description)
        if series_description not in MRI_CANDIDATE_SERIES:
            return f"Unsupported MRI candidate series: {series_description}"
        self.requested_mri_series_description = series_description
        if not self._mri_candidate_row(self.patient_id, series_description):
            return (
                f"Patient {self.patient_id} does not have {series_description}.\n\n"
                + self.mri_availability_text()
            )
        return self.load_patient(self.patient_id)

    def next_patient_with_current_mri_series(self) -> str:
        return self._neighbor_patient_with_mri_series(forward=True)

    def previous_patient_with_current_mri_series(self) -> str:
        return self._neighbor_patient_with_mri_series(forward=False)

    def _neighbor_patient_with_mri_series(self, forward: bool) -> str:
        if not self.queue_rows:
            return "No work queue is loaded."
        description = self.requested_mri_series_description
        indices = [
            index
            for index, row in enumerate(self.queue_rows)
            if self._mri_candidate_row(str(row.get("patient_id", "")), description)
        ]
        if not indices:
            return f"No queued patients have MRI series: {description}"
        current = self.queue_index if self.queue_index is not None else (-1 if forward else len(self.queue_rows))
        if forward:
            candidates = [index for index in indices if index > current]
            if not candidates:
                return f"Already at the last queued patient with {description}."
            next_index = candidates[0]
        else:
            candidates = [index for index in indices if index < current]
            if not candidates:
                return f"Already at the first queued patient with {description}."
            next_index = candidates[-1]
        return self.load_patient(str(self.queue_rows[next_index]["patient_id"]))

    def mri_availability_text(self) -> str:
        rows = self._mri_candidate_rows_for_patient(self.patient_id)
        lines = [
            f"patient={self.patient_id}",
            f"current={self.mri_series_description}",
            f"selected={self.requested_mri_series_description}",
        ]
        for description in MRI_CANDIDATE_SERIES:
            row = rows.get(description)
            short_name = MRI_CANDIDATE_SHORT_NAMES.get(description, description)
            if row:
                current = " <==" if description == self.mri_series_description else ""
                lines.append(
                    f"{short_name}: yes, slices={row.get('instance_count')}, "
                    f"study={row.get('study_folder')}{current}"
                )
            else:
                lines.append(f"{short_name}: no")
        return "\n".join(lines)

    def _mri_candidate_summary(self) -> str:
        rows = self._mri_candidate_rows_for_patient(self.patient_id)
        parts = []
        for description in MRI_CANDIDATE_SERIES:
            short_name = MRI_CANDIDATE_SHORT_NAMES.get(description, description)
            row = rows.get(description)
            if row:
                suffix = "*" if description == self.mri_series_description else ""
                parts.append(f"{short_name}:{row.get('instance_count')}{suffix}")
            else:
                parts.append(f"{short_name}:-")
        return ", ".join(parts)

    def _load_ct_volume(self, patient_id: str, queue_row: dict[str, str] | None) -> Any:
        if queue_row and queue_row.get("ct_series_uid"):
            return load_series_volume(selection_from_queue_row(queue_row, "CT"))
        return load_series_volume(select_series(self.manifest_csv, patient_id, "CT", self.ct_series_description))

    def _load_mri_volume(
        self,
        patient_id: str,
        series_description: str,
        queue_row: dict[str, str] | None,
    ) -> Any:
        row = self._mri_candidate_row(patient_id, series_description)
        if row:
            return load_series_volume(selection_from_manifest_row(row, "MR"))
        if (
            queue_row
            and queue_row.get("mri_series_description") == series_description
            and queue_row.get("mri_series_uid")
        ):
            return load_series_volume(selection_from_queue_row(queue_row, "MRI"))
        raise RuntimeError(f"Patient {patient_id} has no MRI series: {series_description}")

    def _mri_candidate_row(self, patient_id: str, series_description: str) -> dict[str, str] | None:
        return self._mri_candidate_rows_for_patient(patient_id).get(series_description)

    def _mri_candidate_rows_for_patient(self, patient_id: str) -> dict[str, dict[str, str]]:
        patient_id = str(patient_id)
        cached = self._mri_candidate_cache.get(patient_id)
        if cached is not None:
            return cached
        candidates: dict[str, dict[str, str]] = {}
        for row in self.manifest_rows:
            if str(row.get("patient_id", "")) != patient_id:
                continue
            if str(row.get("modality", "")).upper() != "MR":
                continue
            description = str(row.get("series_description", ""))
            if description not in MRI_CANDIDATE_SERIES:
                continue
            previous = candidates.get(description)
            if previous is None or self._series_sort_key(row) > self._series_sort_key(previous):
                candidates[description] = row
        self._mri_candidate_cache[patient_id] = candidates
        return candidates

    @staticmethod
    def _series_sort_key(row: dict[str, str]) -> tuple[int, str, str]:
        try:
            instance_count = int(float(row.get("instance_count") or 0))
        except ValueError:
            instance_count = 0
        return (instance_count, str(row.get("study_folder") or ""), str(row.get("series_uid") or ""))

    def _initialization_output_dir(self, patient_id: str, mri_series_description: str) -> Path:
        return self.db_path.parent / str(patient_id) / "initialization" / self._safe_path_component(
            mri_series_description
        )

    @staticmethod
    def _safe_path_component(value: str) -> str:
        safe = "".join(ch if ch.isalnum() else "_" for ch in value)
        while "__" in safe:
            safe = safe.replace("__", "_")
        return safe.strip("_") or "series"

    def _replace_patient_layers(self) -> None:
        if self.viewer is None or self.ct_volume is None or self.mri_volume is None:
            return
        for name in list(PATIENT_LAYER_NAMES):
            self._remove_layer(name)
        self._replace_image_layer("CT_soft", ct_window(self.ct_volume.array_zyx, -100, 200))
        self._replace_image_layer("CT_bone", ct_window(self.ct_volume.array_zyx, -500, 1500), visible=False)
        self._replace_image_layer("MRI_selected", normalize_percentile(self.mri_volume.array_zyx), visible=False)
        self.ct_points = self._replace_points_layer(
            "CT_landmarks",
            np.empty((0, 3)),
            size=6,
            face_color="yellow",
        )
        self.mri_points = self._replace_points_layer(
            "MRI_landmarks",
            np.empty((0, 3)),
            size=6,
            face_color="cyan",
        )
        self._set_points_layer_add_mode(self.ct_points)
        self._set_points_layer_add_mode(self.mri_points)
        self.ct_points.events.data.connect(lambda event: self._on_points_changed("CT"))
        self.mri_points.events.data.connect(lambda event: self._on_points_changed("MRI"))

    def _activate_points_layer(self, modality: str) -> None:
        if not hasattr(self, "viewer") or self.viewer is None:
            return
        layer = self.ct_points if modality == "CT" else self.mri_points
        if layer is not None:
            self.viewer.layers.selection.active = layer
            self._set_points_layer_add_mode(layer)

    @staticmethod
    def _set_points_layer_add_mode(layer: Any) -> None:
        try:
            layer.mode = "add"
        except Exception:
            pass

    def _show_modality_for_annotation(self, modality: str) -> None:
        if modality == "MRI":
            self._show_only({"MRI_selected", "MRI_landmarks"})
        else:
            self._show_only({"CT_soft", "CT_landmarks"})

    def _set_annotation_modality(self, modality: str) -> None:
        self.current_modality = modality
        widget = self.annotation_state_widget
        if widget is not None:
            try:
                target_widget = widget["target_modality"]
                if target_widget.value != modality:
                    target_widget.value = modality
            except Exception:
                pass
        self._activate_points_layer(modality)
        self._refresh_point_completion_panel()

    def _set_mri_series_widget_value(self, series_description: str) -> None:
        widget = self.mri_series_widget
        if widget is not None:
            try:
                series_widget = widget["mri_series"]
                if series_widget.value != series_description:
                    series_widget.value = series_description
            except Exception:
                pass

    def _set_current_label(self, label: str) -> None:
        if label not in LANDMARK_LABELS:
            return
        self.current_label = label
        widget = self.annotation_state_widget
        if widget is not None:
            try:
                label_widget = widget["label"]
                if label_widget.value != label:
                    label_widget.value = label
            except Exception:
                pass
        self._refresh_point_completion_panel()
        self._notify_status(f"Active landmark: {label}")

    def _refresh_point_completion_panel(self) -> None:
        if self.point_completion_panel is not None:
            self.point_completion_panel.refresh()

    def _show_ct_soft_view(self) -> None:
        self._set_annotation_modality("CT")
        self._show_only({"CT_soft", "CT_landmarks"})

    def _show_ct_bone_view(self) -> None:
        self._set_annotation_modality("CT")
        self._show_only({"CT_bone", "CT_landmarks"})

    def _show_mri_view(self) -> None:
        self._set_annotation_modality("MRI")
        self._show_only({"MRI_selected", "MRI_landmarks"})

    def _show_overlay_view(self) -> None:
        self._set_annotation_modality("MRI")
        self._show_only({"MRI_selected", "MRI_landmarks", "CT_soft_on_MRI_init", "CT_landmarks_projected_to_MRI_init"})
        self._set_dims_order((0, 1, 2), "native MRI overlay")

    def _show_axial_overlay_view(self) -> None:
        if not self._layer_exists("MRI_selected_axial_recon"):
            self._notify_status("Run Compute init + overlay before opening the axial MPR overlay.")
            return
        self._set_annotation_modality("MRI")
        self._show_only(
            {
                "MRI_selected_axial_recon",
                "CT_soft_on_MRI_axial_init",
                "CT_landmarks_projected_to_MRI_axial_init",
            }
        )
        self._set_dims_order((0, 1, 2), "axial MPR overlay")
        self._select_layer("MRI_selected_axial_recon")

    def _set_dims_order(self, order: tuple[int, int, int], name: str) -> None:
        if self.viewer is None:
            return
        self.viewer.dims.order = order
        self._notify_status(f"View orientation: {name}")

    def _select_layer(self, name: str) -> None:
        if self.viewer is None:
            return
        try:
            self.viewer.layers.selection.active = self.viewer.layers[name]
        except Exception:
            pass

    def _layer_exists(self, name: str) -> bool:
        if self.viewer is None:
            return False
        return any(layer.name == name for layer in self.viewer.layers)

    def _restore_saved_points_after_rejected_click(self) -> None:
        self._updating_layers = True
        try:
            self._refresh_layer("CT", self.ct_points)
            self._refresh_layer("MRI", self.mri_points)
        finally:
            self._updating_layers = False

    def _notify_status(self, message: str) -> None:
        if self.viewer is not None:
            try:
                self.viewer.status = message
            except Exception:
                pass
        print(message)

    def status_text(self) -> str:
        records = self.current_landmark_records()
        ct_count = sum(1 for record in records if record["modality"] == "CT")
        mri_count = sum(1 for record in records if record["modality"] == "MRI")
        usable = usable_landmark_pairs(records)
        lr_checks = left_right_lps_checks(records)
        lr_failed = [item for item in lr_checks if not item["pass"]]
        lines = [
            f"patient={self.patient_id}",
            f"queue={self._queue_position_text()}",
            f"CT={self.ct_series_description}",
            f"MRI={self.mri_series_description}",
            f"MRI candidates: {self._mri_candidate_summary()}",
            f"saved CT={ct_count}, MRI={mri_count}",
            f"usable CT/MRI pairs={len(usable)}: {', '.join(usable) if usable else 'none'}",
        ]
        if lr_checks:
            lines.append(f"L/R checks={len(lr_checks)}, failed={len(lr_failed)}")
        else:
            lines.append("L/R checks=not enough paired left/right labels")
        if len(usable) < 3:
            lines.append("Need at least 3 paired labels for rigid initialization.")
        return "\n".join(lines)

    def compute_initialization_overlay(self) -> str:
        if self.viewer is None or self.ct_volume is None or self.mri_volume is None:
            return "Viewer or volumes are not ready."

        records = self.current_landmark_records()
        try:
            result = estimate_rigid_initializer(records, fixed_modality="MRI", moving_modality="CT")
        except Exception as exc:
            return f"Initialization failed: {exc}\n\n{self.status_text()}"

        save_initialization_result(result, self.output_dir)
        resampled_ct = write_resampled_moving_to_fixed(
            moving_image=self.ct_volume.image,
            fixed_image=self.mri_volume.image,
            transform_fixed_to_moving=result.transform_fixed_to_moving,
            output_path=self.output_dir / "ct_resampled_to_mri_init.nii.gz",
            default_value=-1024.0,
        )
        resampled_arr = self._sitk_array(resampled_ct)
        self._replace_image_layer(
            "CT_soft_on_MRI_init",
            ct_window(resampled_arr, -100, 200),
            visible=True,
            opacity=0.55,
            blending="additive",
        )
        self._replace_image_layer(
            "CT_bone_on_MRI_init",
            ct_window(resampled_arr, -500, 1500),
            visible=False,
            opacity=0.55,
            blending="additive",
        )
        mri_axial = self._orient_image_to_lps(self.mri_volume.image)
        ct_axial = self._orient_image_to_lps(resampled_ct)
        self._write_sitk_image(mri_axial, self.output_dir / "mri_selected_lps_axial_mpr.nii.gz")
        self._write_sitk_image(ct_axial, self.output_dir / "ct_resampled_to_mri_init_lps_axial_mpr.nii.gz")
        axial_scale = self._napari_scale_zyx(mri_axial)
        ct_axial_arr = self._sitk_array(ct_axial)
        self._replace_image_layer(
            "MRI_selected_axial_recon",
            normalize_percentile(self._sitk_array(mri_axial)),
            visible=False,
            scale=axial_scale,
        )
        self._replace_image_layer(
            "CT_soft_on_MRI_axial_init",
            ct_window(ct_axial_arr, -100, 200),
            visible=False,
            opacity=0.55,
            blending="additive",
            scale=axial_scale,
        )
        self._replace_image_layer(
            "CT_bone_on_MRI_axial_init",
            ct_window(ct_axial_arr, -500, 1500),
            visible=False,
            opacity=0.55,
            blending="additive",
            scale=axial_scale,
        )
        self._project_ct_landmarks_to_mri_layer(records, result)
        self._project_ct_landmarks_to_image_layer(
            records=records,
            result=result,
            target_image=mri_axial,
            layer_name="CT_landmarks_projected_to_MRI_axial_init",
            visible=False,
            scale=axial_scale,
        )
        self._focus_overlay_layers()
        summary = result_summary(result)
        residual_lines = [
            f"{item['label']}: {float(item['residual_mm']):.2f} mm" for item in summary["residuals"]
        ]
        return (
            f"status={summary['status']}\n"
            f"n={summary['n_landmarks_used']}\n"
            f"median={summary['median_residual_mm']:.2f} mm, max={summary['max_residual_mm']:.2f} mm\n"
            f"saved={self.output_dir}\n"
            "press 4 for native MRI overlay, 5 for axial MPR overlay\n"
            + "\n".join(residual_lines)
        )

    def _project_ct_landmarks_to_mri_layer(self, records: list[dict[str, Any]], result: Any) -> None:
        if self.mri_volume is None:
            return
        self._project_ct_landmarks_to_image_layer(
            records=records,
            result=result,
            target_image=self.mri_volume.image,
            layer_name="CT_landmarks_projected_to_MRI_init",
            visible=True,
        )

    def _project_ct_landmarks_to_image_layer(
        self,
        records: list[dict[str, Any]],
        result: Any,
        target_image: Any,
        layer_name: str,
        visible: bool,
        scale: tuple[float, float, float] | None = None,
    ) -> None:
        by_key = {
            (record["modality"], record["landmark_label"]): record
            for record in records
            if record.get("use_for_transform", True)
        }
        points = []
        labels = []
        residuals = []
        for residual in result.residuals:
            label = str(residual["label"])
            ct_record = by_key.get(("CT", label))
            if not ct_record:
                continue
            ct_lps = tuple(float(v) for v in ct_record["physical_lps_mm"])
            projected_lps = result.transform_moving_to_fixed.TransformPoint(ct_lps)
            points.append(napari_zyx_from_physical_lps(target_image, projected_lps))
            labels.append(label)
            residuals.append(float(residual["residual_mm"]))
        kwargs: dict[str, Any] = {}
        if scale is not None:
            kwargs["scale"] = scale
        self._replace_points_layer(
            layer_name,
            np.asarray(points, dtype=float) if points else np.empty((0, 3)),
            face_color="red",
            size=7,
            features={"label": labels, "residual_mm": residuals},
            visible=visible,
            **kwargs,
        )

    def _focus_overlay_layers(self) -> None:
        if self.viewer is None:
            return
        visible_names = {"MRI_selected", "MRI_landmarks", "CT_soft_on_MRI_init", "CT_landmarks_projected_to_MRI_init"}
        hidden_names = {"CT_soft", "CT_bone", "CT_landmarks"}
        for layer in list(self.viewer.layers):
            if layer.name in visible_names:
                layer.visible = True
            if layer.name in hidden_names:
                layer.visible = False

    def _bind_keys(self) -> None:
        if self.viewer is None:
            return
        self._bind_workbench_shortcuts_to_target(self.viewer)

    def _bind_workbench_shortcuts_to_target(self, target: Any) -> None:
        if target is None or not hasattr(target, "bind_key"):
            return

        def bind(key: str, callback: Any) -> None:
            def handler(_owner: Any) -> None:
                callback()

            try:
                target.bind_key(key, handler, overwrite=True)
            except Exception:
                pass

        bind("1", self._show_ct_soft_view)
        bind("2", self._show_ct_bone_view)
        bind("3", self._show_mri_view)
        bind("4", self._show_overlay_view)
        bind("5", self._show_axial_overlay_view)
        bind("n", self.next_patient)
        bind("p", self.previous_patient)
        bind("x", lambda: self._set_dims_order((0, 1, 2), "axial"))
        bind("c", lambda: self._set_dims_order((1, 0, 2), "coronal"))
        bind("v", lambda: self._set_dims_order((2, 0, 1), "sagittal"))
        for key, label in LANDMARK_SHORTCUTS.items():
            bind(key, lambda label=label: self._set_current_label(label))

    def _show_only(self, visible_names: set[str]) -> None:
        if self.viewer is None:
            return
        for layer in self.viewer.layers:
            layer.visible = layer.name in visible_names

    def _replace_image_layer(self, name: str, data: np.ndarray, **kwargs: Any) -> Any:
        if self.viewer is None:
            return None
        self._remove_layer(name)
        layer = self.viewer.add_image(data, name=name, **kwargs)
        self._bind_workbench_shortcuts_to_target(layer)
        return layer

    def _replace_points_layer(self, name: str, data: np.ndarray, **kwargs: Any) -> Any:
        if self.viewer is None:
            return None
        self._remove_layer(name)
        layer = self.viewer.add_points(data, name=name, **kwargs)
        self._bind_workbench_shortcuts_to_target(layer)
        return layer

    def _remove_layer(self, name: str) -> None:
        if self.viewer is None:
            return
        for layer in list(self.viewer.layers):
            if layer.name == name:
                self.viewer.layers.remove(layer)
                return

    def _queue_row_for_patient(self, patient_id: str) -> dict[str, str] | None:
        for row in self.queue_rows:
            if row.get("patient_id") == str(patient_id):
                return row
        return None

    def _queue_index_for_patient(self, patient_id: str) -> int | None:
        for index, row in enumerate(self.queue_rows):
            if row.get("patient_id") == str(patient_id):
                return index
        return None

    def _queue_position_text(self) -> str:
        if not self.queue_rows:
            return "none"
        if self.queue_index is None:
            return f"not in queue / {len(self.queue_rows)}"
        return f"{self.queue_index + 1}/{len(self.queue_rows)}"

    @staticmethod
    def _read_work_queue(path: Path | None) -> list[dict[str, str]]:
        if not path or not path.exists():
            return []
        with path.open(newline="", encoding="utf-8-sig") as f:
            return [dict(row) for row in csv.DictReader(f)]

    def _resize_viewer_window(self) -> None:
        if self.viewer is None or self.window_scale <= 0:
            return
        qt_window = getattr(self.viewer.window, "_qt_window", None)
        if qt_window is None:
            return
        scale = max(float(self.window_scale), 0.1)
        current = qt_window.size()
        width = max(int(current.width() * scale), 1400)
        height = max(int(current.height() * scale), 1000)
        screen = qt_window.screen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(width, max(int(available.width() * 0.98), 800))
            height = min(height, max(int(available.height() * 0.95), 700))
        qt_window.resize(width, height)
        qt_window.move(20, 20)

    @staticmethod
    def _sitk_array(image: Any) -> np.ndarray:
        import SimpleITK as sitk

        return sitk.GetArrayFromImage(image)

    @staticmethod
    def _orient_image_to_lps(image: Any) -> Any:
        import SimpleITK as sitk

        return sitk.DICOMOrient(image, "LPS")

    @staticmethod
    def _napari_scale_zyx(image: Any) -> tuple[float, float, float]:
        spacing_xyz = tuple(float(v) for v in image.GetSpacing())
        return (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

    @staticmethod
    def _write_sitk_image(image: Any, path: str | Path) -> None:
        import SimpleITK as sitk

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, str(out))


def run(
    manifest_csv: str | Path,
    patient_id: str,
    db_path: str | Path,
    ct_series_description: str = "AX",
    mri_series_description: str = "T2 COR dixon_(IN W)_in",
    work_queue_csv: str | Path | None = None,
    window_scale: float = 2.0,
    annotator_id: str = "default",
) -> None:
    app = LandmarkWorkbench(
        manifest_csv=manifest_csv,
        patient_id=patient_id,
        db_path=db_path,
        ct_series_description=ct_series_description,
        mri_series_description=mri_series_description,
        work_queue_csv=work_queue_csv,
        window_scale=window_scale,
        annotator_id=annotator_id,
    )
    app.launch()
