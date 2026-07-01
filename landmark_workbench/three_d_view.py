from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


MODE_VIEW = "View"
MODE_PITCH = "Pitch"
MODE_SCALE_X = "Scale X"
MODE_SCALE_Y = "Scale Y"
MODE_SCALE_Z = "Scale Z"
MODES = (MODE_VIEW, MODE_PITCH, MODE_SCALE_X, MODE_SCALE_Y, MODE_SCALE_Z)


@dataclass
class PointCloud3D:
    points: np.ndarray
    color: tuple[int, int, int, int]
    point_size: int = 2


@dataclass
class LineSet3D:
    segments: np.ndarray
    color: tuple[int, int, int, int]
    width: int = 1


@dataclass
class Marker3D:
    point: np.ndarray
    color: tuple[int, int, int, int]
    label: str


class Globe3DPanel:
    """Software-rendered 3D QC panel using DICOM/SimpleITK LPS mm geometry."""

    def __init__(self, workbench: Any):
        from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

        self.workbench = workbench
        self.widget = QWidget()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(list(MODES))
        self.status_label = QLabel("3D panel not ready")
        self.status_label.setWordWrap(True)
        self.refresh_button = QPushButton("Refresh 3D")
        self.canvas = Software3DCanvas(self)
        self.scene_origin_lps: np.ndarray | None = None
        self.point_clouds: list[PointCloud3D] = []
        self.line_sets: list[LineSet3D] = []
        self.markers: list[Marker3D] = []
        self.scene_bounds: tuple[np.ndarray, np.ndarray] | None = None
        self._last_scene_key: tuple[str, str, str] | None = None

        controls = QHBoxLayout()
        controls.addWidget(QLabel("3D mode"))
        controls.addWidget(self.mode_combo)
        controls.addWidget(self.refresh_button)
        controls.addWidget(self.status_label, stretch=1)

        layout = QVBoxLayout(self.widget)
        layout.addLayout(controls)
        layout.addWidget(self.canvas, stretch=1)

        self.mode_combo.currentTextChanged.connect(lambda _: self.update_interaction_mode())
        self.refresh_button.clicked.connect(self.update_scene)
        self.update_interaction_mode()

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            return
        if self.mode_combo.currentText() != mode:
            self.mode_combo.setCurrentText(mode)
        self.update_interaction_mode()

    def mode(self) -> str:
        return str(self.mode_combo.currentText())

    def update_interaction_mode(self) -> None:
        if self.mode() == MODE_VIEW:
            self.status_label.setText("View mode: drag to rotate, wheel to zoom")
        else:
            self.status_label.setText(f"{self.mode()} mode: wheel adjusts MRI transform; camera angle is kept")
        self.canvas.update()

    def adjust_current_parameter(self, notches: float) -> None:
        mode = self.mode()
        if mode == MODE_PITCH and self.workbench.pitch_spin is not None:
            self._step_spin(self.workbench.pitch_spin, notches * 0.25)
        elif mode == MODE_SCALE_X and self.workbench.scale_x_spin is not None:
            self._step_spin(self.workbench.scale_x_spin, notches * 0.005)
        elif mode == MODE_SCALE_Y and self.workbench.scale_y_spin is not None:
            self._step_spin(self.workbench.scale_y_spin, notches * 0.005)
        elif mode == MODE_SCALE_Z and self.workbench.scale_z_spin is not None:
            self._step_spin(self.workbench.scale_z_spin, notches * 0.005)
        self.update_status_with_parameters()

    @staticmethod
    def _step_spin(spin: Any, delta: float) -> None:
        value = float(spin.value()) + float(delta)
        value = max(float(spin.minimum()), min(float(spin.maximum()), value))
        spin.setValue(value)

    def update_status_with_parameters(self) -> None:
        pitch = self.workbench.pitch_degrees()
        sx, sy, sz = self.workbench.manual_scale_xyz()
        self.status_label.setText(
            f"{self.mode()} | pitch={pitch:.2f}, scale=({sx:.3f}, {sy:.3f}, {sz:.3f})"
        )

    def update_scene(self) -> None:
        if self.workbench.ct_lps is None or self.workbench.mri_lps is None:
            self.status_label.setText("Load CT/MRI before opening 3D scene")
            return
        self.point_clouds.clear()
        self.line_sets.clear()
        self.markers.clear()
        try:
            scene_key = self.current_scene_key()
            if scene_key != self._last_scene_key:
                self.canvas.reset_to_default_view()
                self._last_scene_key = scene_key
            self.scene_origin_lps = self.compute_scene_origin_lps()
            self.add_volume_clouds()
            self.add_bounding_boxes()
            self.add_globe_geometry()
            self.update_scene_bounds()
            self.update_status_with_parameters()
            self.canvas.update()
        except Exception as exc:
            self.status_label.setText(f"3D render failed: {exc}")

    def current_scene_key(self) -> tuple[str, str, str]:
        ct_uid = getattr(getattr(self.workbench, "ct_volume", None), "selection", None)
        mri_uid = getattr(getattr(self.workbench, "mri_volume", None), "selection", None)
        return (
            str(getattr(self.workbench, "patient_id", "")),
            str(getattr(ct_uid, "series_uid", "")),
            str(getattr(mri_uid, "series_uid", "")),
        )

    def compute_scene_origin_lps(self) -> np.ndarray:
        ct_center = self.image_bbox_center_lps(self.workbench.ct_lps)
        mri_center = self.image_bbox_center_lps(self.displayed_mri_image())
        return (ct_center + mri_center) * 0.5

    def displayed_mri_image(self) -> Any:
        if self.workbench.globe_mri_on_ct_lps is not None:
            return self.workbench.globe_mri_on_ct_lps
        return self.workbench.mri_lps

    def add_volume_clouds(self) -> None:
        self.add_volume_cloud(
            array=self.workbench.ct_bone_lps,
            image=self.workbench.ct_lps,
            threshold=0.38,
            max_points=14000,
            color=(255, 198, 86, 95),
        )
        if self.workbench.globe_mri_on_ct_lps is not None:
            mri_image = self.workbench.globe_mri_on_ct_lps
            mri_array = self.workbench.globe_mri_on_ct_display_lps
        else:
            mri_image = self.workbench.mri_lps
            mri_array = self.workbench.mri_display_lps
        self.add_volume_cloud(
            array=mri_array,
            image=mri_image,
            threshold=0.42,
            max_points=14000,
            color=(38, 220, 255, 82),
        )

    def add_volume_cloud(
        self,
        array: np.ndarray | None,
        image: Any,
        threshold: float,
        max_points: int,
        color: tuple[int, int, int, int],
    ) -> None:
        if array is None:
            return
        stride_zyx = self.volume_stride(array.shape, max_dim=115)
        sampled = array[:: stride_zyx[0], :: stride_zyx[1], :: stride_zyx[2]]
        coords_zyx = np.argwhere(sampled > float(threshold))
        if coords_zyx.size == 0:
            flat = sampled.reshape(-1)
            if flat.size == 0:
                return
            cutoff = np.percentile(flat, 97.5)
            coords_zyx = np.argwhere(sampled >= cutoff)
        if coords_zyx.shape[0] > max_points:
            values = sampled[coords_zyx[:, 0], coords_zyx[:, 1], coords_zyx[:, 2]]
            keep = np.argpartition(values, -int(max_points))[-int(max_points):]
            coords_zyx = coords_zyx[keep]
        points = self.indices_zyx_to_scene_lps(image, coords_zyx, stride_zyx)
        if points.size:
            self.point_clouds.append(PointCloud3D(points=points.astype(np.float32), color=color, point_size=2))

    @staticmethod
    def volume_stride(shape: tuple[int, ...], max_dim: int) -> tuple[int, int, int]:
        return tuple(max(1, int(np.ceil(float(dim) / float(max_dim)))) for dim in shape[:3])

    def indices_zyx_to_scene_lps(
        self,
        image: Any,
        coords_zyx: np.ndarray,
        stride_zyx: tuple[int, int, int],
    ) -> np.ndarray:
        coords = np.asarray(coords_zyx, dtype=np.float64)
        if coords.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        index_xyz = np.column_stack(
            [
                coords[:, 2] * float(stride_zyx[2]),
                coords[:, 1] * float(stride_zyx[1]),
                coords[:, 0] * float(stride_zyx[0]),
            ]
        )
        origin = np.asarray(image.GetOrigin(), dtype=float)
        spacing = np.asarray(image.GetSpacing(), dtype=float)
        direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
        physical = origin[None, :] + (index_xyz * spacing[None, :]) @ direction.T
        return self.lps_to_scene(physical)

    def add_bounding_boxes(self) -> None:
        self.add_bounding_box(self.workbench.ct_lps, (255, 209, 102, 235), width=2)
        self.add_bounding_box(self.displayed_mri_image(), (34, 211, 238, 235), width=2)

    def add_bounding_box(self, image: Any, color: tuple[int, int, int, int], width: int) -> None:
        corners = self.lps_to_scene(self.image_bbox_lps(image))
        edges = [
            (0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7),
            (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7),
        ]
        self.line_sets.append(
            LineSet3D(
                segments=np.asarray([[corners[start], corners[end]] for start, end in edges], dtype=np.float32),
                color=color,
                width=width,
            )
        )

    def add_globe_geometry(self) -> None:
        fits = self.workbench.current_globe_sphere_fits()
        for modality, sphere_color, marker_color in (
            ("CT", (255, 242, 0, 230), (255, 255, 80, 255)),
            ("MRI", (0, 245, 255, 230), (40, 255, 255, 255)),
        ):
            centers = []
            for side in ("L", "R"):
                fit = fits.get((modality, side))
                if fit is None:
                    continue
                center = np.asarray(fit.center_lps, dtype=float)
                if modality == "MRI" and self.workbench.globe_registration_result is not None:
                    center = np.asarray(
                        self.workbench.globe_registration_result.transform_moving_to_fixed.TransformPoint(tuple(center)),
                        dtype=float,
                    )
                centers.append(center)
                self.add_sphere_wireframe(center, float(fit.radius_mm), sphere_color, width=2)
                self.markers.append(
                    Marker3D(
                        point=self.lps_to_scene(center).astype(np.float32),
                        color=marker_color,
                        label=f"{modality} {side}C",
                    )
                )
            if len(centers) == 2:
                self.line_sets.append(
                    LineSet3D(
                        segments=np.asarray(
                            [[self.lps_to_scene(centers[0]), self.lps_to_scene(centers[1])]],
                            dtype=np.float32,
                        ),
                        color=(255, 255, 255, 225),
                        width=3,
                    )
                )

    def add_sphere_wireframe(
        self,
        center_lps: np.ndarray,
        radius_mm: float,
        color: tuple[int, int, int, int],
        width: int,
    ) -> None:
        center = self.lps_to_scene(np.asarray(center_lps, dtype=float))
        segments: list[np.ndarray] = []
        for axis in range(3):
            circle = self.sphere_circle_points(center, radius_mm, axis=axis, n=96)
            segments.extend(np.stack([circle[:-1], circle[1:]], axis=1))
            segments.append(np.asarray([circle[-1], circle[0]], dtype=np.float32))
        self.line_sets.append(LineSet3D(segments=np.asarray(segments, dtype=np.float32), color=color, width=width))

    @staticmethod
    def sphere_circle_points(center: np.ndarray, radius: float, axis: int, n: int) -> np.ndarray:
        theta = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False, dtype=np.float32)
        points = np.zeros((int(n), 3), dtype=np.float32)
        axes = [0, 1, 2]
        axes.remove(int(axis))
        points[:, axes[0]] = np.cos(theta) * float(radius)
        points[:, axes[1]] = np.sin(theta) * float(radius)
        return points + center.astype(np.float32)[None, :]

    def update_scene_bounds(self) -> None:
        items = []
        for cloud in self.point_clouds:
            if cloud.points.size:
                items.append(cloud.points)
        for lines in self.line_sets:
            if lines.segments.size:
                items.append(lines.segments.reshape(-1, 3))
        for marker in self.markers:
            items.append(marker.point.reshape(1, 3))
        if not items:
            self.scene_bounds = (np.array([-50.0, -50.0, -50.0]), np.array([50.0, 50.0, 50.0]))
            return
        points = np.vstack(items)
        self.scene_bounds = (points.min(axis=0), points.max(axis=0))

    def image_bbox_lps(self, image: Any) -> np.ndarray:
        size = tuple(int(v) for v in image.GetSize())
        corners = []
        for x in (0.0, float(max(size[0] - 1, 0))):
            for y in (0.0, float(max(size[1] - 1, 0))):
                for z in (0.0, float(max(size[2] - 1, 0))):
                    corners.append(image.TransformContinuousIndexToPhysicalPoint((x, y, z)))
        return np.asarray(corners, dtype=float)

    def image_bbox_center_lps(self, image: Any) -> np.ndarray:
        corners = self.image_bbox_lps(image)
        return (corners.min(axis=0) + corners.max(axis=0)) * 0.5

    def lps_to_scene(self, points_lps: np.ndarray) -> np.ndarray:
        points = np.asarray(points_lps, dtype=float)
        if self.scene_origin_lps is None:
            return points
        return points - self.scene_origin_lps


class Software3DCanvas:
    def __new__(cls, panel: Globe3DPanel):
        from qtpy.QtCore import QPointF, Qt
        from qtpy.QtGui import QColor, QPainter, QPen
        from qtpy.QtWidgets import QWidget

        class _Canvas(QWidget):
            def __init__(self, owner: Globe3DPanel):
                super().__init__()
                self.owner = owner
                self.azimuth_deg = 0.0
                self.elevation_deg = 0.0
                self.zoom = 1.0
                self.drag_start: tuple[float, float] | None = None
                self.setMinimumSize(520, 520)
                self.setMouseTracking(True)

            def reset_to_default_view(self) -> None:
                self.azimuth_deg = 0.0
                self.elevation_deg = 0.0
                self.zoom = 1.0
                self.drag_start = None
                self.update()

            def paintEvent(self, event: Any) -> None:
                painter = QPainter(self)
                painter.fillRect(self.rect(), QColor(5, 5, 5))
                painter.setRenderHint(QPainter.RenderHint.Antialiasing if hasattr(QPainter, "RenderHint") else QPainter.Antialiasing, True)
                if self.owner.scene_bounds is None:
                    self._draw_center_text(painter, "Load patient to build 3D QC scene")
                    return
                try:
                    self._draw_scene(painter)
                except Exception as exc:
                    self._draw_center_text(painter, f"3D paint failed: {exc}")

            def _draw_scene(self, painter: Any) -> None:
                scale = self.scene_scale()
                rotation = self.rotation_matrix()
                for cloud in self.owner.point_clouds:
                    self._draw_cloud(painter, cloud, rotation, scale)
                for line_set in self.owner.line_sets:
                    self._draw_line_set(painter, line_set, rotation, scale)
                for marker in self.owner.markers:
                    self._draw_marker(painter, marker, rotation, scale)
                self._draw_legend(painter)

            def _draw_cloud(self, painter: Any, cloud: PointCloud3D, rotation: np.ndarray, scale: float) -> None:
                if cloud.points.size == 0:
                    return
                screen = self.project_points(cloud.points, rotation, scale)
                color = QColor(*cloud.color)
                painter.setPen(QPen(color, int(cloud.point_size)))
                if len(screen) > 18000:
                    step = int(np.ceil(len(screen) / 18000.0))
                    screen = screen[::step]
                for x, y, _z in screen:
                    painter.drawPoint(QPointF(float(x), float(y)))

            def _draw_line_set(self, painter: Any, line_set: LineSet3D, rotation: np.ndarray, scale: float) -> None:
                if line_set.segments.size == 0:
                    return
                color = QColor(*line_set.color)
                painter.setPen(QPen(color, int(line_set.width)))
                flat = line_set.segments.reshape(-1, 3)
                screen = self.project_points(flat, rotation, scale).reshape(-1, 2, 3)
                for start, end in screen:
                    painter.drawLine(QPointF(float(start[0]), float(start[1])), QPointF(float(end[0]), float(end[1])))

            def _draw_marker(self, painter: Any, marker: Marker3D, rotation: np.ndarray, scale: float) -> None:
                screen = self.project_points(marker.point.reshape(1, 3), rotation, scale)[0]
                painter.setPen(QPen(QColor(255, 255, 255), 2))
                painter.setBrush(QColor(*marker.color))
                radius = 6
                painter.drawEllipse(QPointF(float(screen[0]), float(screen[1])), radius, radius)
                painter.drawText(QPointF(float(screen[0] + 8), float(screen[1] - 8)), marker.label)

            def _draw_legend(self, painter: Any) -> None:
                painter.setPen(QPen(QColor(220, 220, 220), 1))
                lines = [
                    "3D QC: CT yellow/orange, MRI cyan | default side view",
                    f"mode={self.owner.mode()} | drag rotates only in View mode",
                    f"az={self.azimuth_deg:.1f}, el={self.elevation_deg:.1f}, zoom={self.zoom:.2f}",
                ]
                y = 18
                for line in lines:
                    painter.drawText(10, y, line)
                    y += 17

            def _draw_center_text(self, painter: Any, text: str) -> None:
                painter.setPen(QPen(QColor(220, 220, 220), 1))
                alignment = Qt.AlignmentFlag.AlignCenter if hasattr(Qt, "AlignmentFlag") else Qt.AlignCenter
                painter.drawText(self.rect(), alignment, text)

            def scene_scale(self) -> float:
                lower, upper = self.owner.scene_bounds
                span = np.maximum(upper - lower, 1.0)
                base = 0.82 * min(max(self.width(), 1), max(self.height(), 1)) / float(np.max(span))
                return base * float(self.zoom)

            def rotation_matrix(self) -> np.ndarray:
                az = np.deg2rad(float(self.azimuth_deg))
                el = np.deg2rad(float(self.elevation_deg))
                cy, sy = np.cos(az), np.sin(az)
                cx, sx = np.cos(el), np.sin(el)
                # Base view maps LPS coordinates to a patient side view:
                # screen x = posterior direction, screen y = superior direction,
                # depth = left-right direction. This makes both globes overlap.
                base_side_view = np.array(
                    [
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0],
                    ],
                    dtype=float,
                )
                ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
                rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
                return rx @ ry @ base_side_view

            def project_points(self, points: np.ndarray, rotation: np.ndarray, scale: float) -> np.ndarray:
                rotated = np.asarray(points, dtype=float) @ rotation.T
                out = np.empty((rotated.shape[0], 3), dtype=float)
                out[:, 0] = self.width() * 0.5 + rotated[:, 0] * scale
                out[:, 1] = self.height() * 0.5 - rotated[:, 1] * scale
                out[:, 2] = rotated[:, 2]
                return out

            def mousePressEvent(self, event: Any) -> None:
                left_button = Qt.MouseButton.LeftButton if hasattr(Qt, "MouseButton") else Qt.LeftButton
                if event.button() == left_button and self.owner.mode() == MODE_VIEW:
                    pos = event.position() if hasattr(event, "position") else event.pos()
                    self.drag_start = (float(pos.x()), float(pos.y()))

            def mouseMoveEvent(self, event: Any) -> None:
                if self.drag_start is None or self.owner.mode() != MODE_VIEW:
                    return
                pos = event.position() if hasattr(event, "position") else event.pos()
                x, y = float(pos.x()), float(pos.y())
                old_x, old_y = self.drag_start
                self.azimuth_deg += (x - old_x) * 0.35
                self.elevation_deg = max(-85.0, min(85.0, self.elevation_deg + (y - old_y) * 0.25))
                self.drag_start = (x, y)
                self.update()

            def mouseReleaseEvent(self, event: Any) -> None:
                self.drag_start = None

            def wheelEvent(self, event: Any) -> None:
                delta = float(event.angleDelta().y())
                if delta == 0.0:
                    return
                notches = delta / 120.0
                if self.owner.mode() == MODE_VIEW:
                    self.zoom = max(0.08, min(25.0, self.zoom * (1.12 ** notches)))
                    self.update()
                else:
                    self.owner.adjust_current_parameter(notches)
                    event.accept()

        return _Canvas(panel)
