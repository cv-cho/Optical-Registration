from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import (
    image_direction_3x3,
    itk_xyz_to_napari_zyx,
    napari_zyx_from_physical_lps,
    point_inside_array_shape_zyx,
)
from .globe import SphereFit, fit_globe_spheres
from .globe_registration import (
    estimate_globe_manual_initializer,
    resample_mri_to_ct,
    save_globe_manual_registration,
)
from .napari_app import MRI_CANDIDATE_SERIES, MRI_CANDIDATE_SHORT_NAMES
from .schema import LANDMARK_LABELS
from .store import AnnotationStore, GlobeCenterOverrideRecord, GlobeSurfacePointRecord, LandmarkRecord
from .three_d_view import Globe3DPanel, MODE_PITCH, MODE_SCALE_X, MODE_SCALE_Y, MODE_SCALE_Z, MODE_VIEW
from .transform import (
    estimate_rigid_initializer,
    resample_moving_to_fixed,
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


VIEW_AXIAL = "Axial"
VIEW_CORONAL = "Coronal"
VIEW_SAGITTAL = "Sagittal"
SOURCE_CT = "CT"
SOURCE_MRI = "MRI"
SOURCE_OVERLAY = "Overlay"
SOURCE_MRI_ON_CT = "MRI on CT"
CLICK_TARGET_GLOBE = "Globe surface"
DEFAULT_MRI_SERIES = "T2 COR dixon_(IN W)_in"
LINE_TOOL_POINT = "Point"
LINE_TOOL_DRAW = "Draw line"
LINE_TOOL_EDIT = "Edit line"
LINE_TOOL_ERASE = "Erase line"
LINE_CONSTRAINT_FREE = "Free"
LINE_CONSTRAINT_HORIZONTAL = "Horizontal"
LINE_CONSTRAINT_VERTICAL = "Vertical"
LINE_CONSTRAINT_ANGLE = "Angle"


@dataclass
class DisplayPoint:
    row: float
    col: float
    color: str
    label: str
    modality: str
    editable: bool = True
    kind: str = "landmark"
    record_id: int | None = None
    side: str | None = None


@dataclass
class DisplayLine:
    row1_norm: float
    col1_norm: float
    row2_norm: float
    col2_norm: float
    color: str = "#7cff6b"


class SliceCanvas:
    def __init__(
        self,
        parent: Any,
        click_callback: Any,
        drag_callback: Any,
        delete_callback: Any,
        step_callback: Any,
        line_tool_callback: Any,
        line_constraint_callback: Any,
        line_angle_callback: Any,
        line_created_callback: Any,
        line_deleted_callback: Any,
        activate_callback: Any,
    ):
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QWidget

        class _Canvas(QWidget):
            def __init__(
                self,
                callback: Any,
                point_drag_callback: Any,
                point_delete_callback: Any,
                wheel_callback: Any,
                guide_tool_callback: Any,
                guide_constraint_callback: Any,
                guide_angle_callback: Any,
                guide_created_callback: Any,
                guide_deleted_callback: Any,
                activate_callback: Any,
            ):
                super().__init__()
                self.callback = callback
                self.point_drag_callback = point_drag_callback
                self.point_delete_callback = point_delete_callback
                self.wheel_callback = wheel_callback
                self.guide_tool_callback = guide_tool_callback
                self.guide_constraint_callback = guide_constraint_callback
                self.guide_angle_callback = guide_angle_callback
                self.guide_created_callback = guide_created_callback
                self.guide_deleted_callback = guide_deleted_callback
                self.activate_callback = activate_callback
                self.rgb: np.ndarray | None = None
                self.points: list[DisplayPoint] = []
                self.lines: list[DisplayLine] = []
                self.orientation_labels: tuple[str, str] | None = None
                self.preview_line: DisplayLine | None = None
                self.image_rect = None
                self.viewport_image_rect: tuple[float, float, float, float] | None = None
                self.overview_rect = None
                self.overview_view_rect = None
                self.zoom_factor = 1.0
                self.zoom_center: tuple[float, float] | None = None
                self.max_zoom_factor = 12.0
                self.drag_point: DisplayPoint | None = None
                self.drag_offset: tuple[float, float] = (0.0, 0.0)
                self.drag_overview = False
                self.drag_pan = False
                self.pan_previous_pos: tuple[float, float] | None = None
                self.active_line: DisplayLine | None = None
                self.active_line_part: str | None = None
                self.line_anchor: tuple[float, float] | None = None
                self.line_drag_previous: tuple[float, float] | None = None
                self.setMinimumSize(512, 512)
                self.setMouseTracking(True)
                self.update_cursor()

            def set_scene(
                self,
                rgb: np.ndarray,
                points: list[DisplayPoint],
                lines: list[DisplayLine],
                orientation_labels: tuple[str, str] | None = None,
            ) -> None:
                old_shape = self.rgb.shape[:2] if self.rgb is not None else None
                self.rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                self.points = points
                self.lines = lines
                self.orientation_labels = orientation_labels
                if old_shape is not None and old_shape != self.rgb.shape[:2]:
                    self.reset_zoom(update=False)
                else:
                    self._clamp_zoom_center()
                self.update()

            def paintEvent(self, event: Any) -> None:
                from qtpy.QtCore import QRectF, Qt
                from qtpy.QtGui import QColor, QImage, QPainter, QPen

                painter = QPainter(self)
                painter.fillRect(self.rect(), QColor(15, 15, 15))
                self.image_rect = None
                self.viewport_image_rect = None
                self.overview_rect = None
                self.overview_view_rect = None
                if self.rgb is None:
                    return
                h, w, _ = self.rgb.shape
                qformat = getattr(QImage, "Format_RGB888", None)
                if hasattr(QImage, "Format"):
                    qformat = QImage.Format.Format_RGB888
                qimage = QImage(self.rgb.data, w, h, 3 * w, qformat)
                scale = min(self.width() / max(w, 1), self.height() / max(h, 1))
                draw_w = w * scale
                draw_h = h * scale
                left = (self.width() - draw_w) / 2.0
                top = (self.height() - draw_h) / 2.0
                rect = QRectF(left, top, draw_w, draw_h)
                self.image_rect = rect
                viewport = self._zoom_viewport()
                self.viewport_image_rect = viewport
                row0, col0, row1, col1 = viewport
                painter.drawImage(rect, qimage, QRectF(col0, row0, col1 - col0, row1 - row0))
                painter.save()
                painter.setClipRect(rect)
                for line in self.lines:
                    self._paint_line(painter, line, selected=line is self.active_line)
                if self.preview_line is not None:
                    pen = QPen(QColor("#f7f871"), 2)
                    pen.setStyle(Qt.PenStyle.DashLine if hasattr(Qt, "PenStyle") else Qt.DashLine)
                    painter.setPen(pen)
                    r1, c1, r2, c2 = self._line_to_image(self.preview_line)
                    x1, y1 = self._image_to_widget(r1, c1)
                    x2, y2 = self._image_to_widget(r2, c2)
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))
                for point in self.points:
                    x, y = self._image_to_widget(point.row, point.col)
                    painter.setPen(QPen(QColor(point.color), 2))
                    radius = 6 if point.editable else 4
                    painter.drawEllipse(int(x - radius), int(y - radius), radius * 2, radius * 2)
                    painter.drawText(int(x + 7), int(y - 7), point.label)
                painter.restore()
                self._paint_orientation_labels(painter, left, top, draw_w)
                self._paint_zoom_overview(painter, qimage, rect, viewport)

            def _paint_orientation_labels(self, painter: Any, left: float, top: float, draw_w: float) -> None:
                if not self.orientation_labels:
                    return
                from qtpy.QtCore import QRectF, Qt
                from qtpy.QtGui import QColor, QFont

                left_label, right_label = self.orientation_labels
                font = QFont()
                font.setBold(True)
                font.setPointSize(18)
                painter.setFont(font)
                align_left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop if hasattr(Qt, "AlignmentFlag") else Qt.AlignLeft | Qt.AlignTop
                align_right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop if hasattr(Qt, "AlignmentFlag") else Qt.AlignRight | Qt.AlignTop
                label_width = 80.0
                label_height = 36.0
                margin = 10.0
                items = [
                    (left_label, QRectF(left + margin, top + margin, label_width, label_height), align_left),
                    (right_label, QRectF(left + draw_w - label_width - margin, top + margin, label_width, label_height), align_right),
                ]
                for label, rect, alignment in items:
                    painter.setPen(QColor(0, 0, 0, 220))
                    for dx, dy in [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)]:
                        painter.drawText(rect.translated(dx, dy), alignment, label)
                    painter.setPen(QColor(255, 255, 255))
                    painter.drawText(rect, alignment, label)

            def _paint_zoom_overview(self, painter: Any, qimage: Any, image_rect: Any, viewport: tuple[float, float, float, float]) -> None:
                if self.rgb is None or self.zoom_factor <= 1.0:
                    return
                from qtpy.QtCore import QRectF
                from qtpy.QtGui import QColor, QPen

                h, w, _ = self.rgb.shape
                max_size = min(float(image_rect.width()), float(image_rect.height())) * 0.28
                overview_w = max(110.0, min(180.0, max_size))
                overview_h = overview_w * float(h) / max(float(w), 1.0)
                if overview_h > 180.0:
                    overview_h = 180.0
                    overview_w = overview_h * float(w) / max(float(h), 1.0)
                margin = 10.0
                top_offset = 48.0 if self.orientation_labels else margin
                overview = QRectF(float(image_rect.left()) + margin, float(image_rect.top()) + top_offset, overview_w, overview_h)
                self.overview_rect = overview
                row0, col0, row1, col1 = viewport
                view_rect = QRectF(
                    overview.left() + col0 / max(float(w), 1.0) * overview.width(),
                    overview.top() + row0 / max(float(h), 1.0) * overview.height(),
                    (col1 - col0) / max(float(w), 1.0) * overview.width(),
                    (row1 - row0) / max(float(h), 1.0) * overview.height(),
                )
                self.overview_view_rect = view_rect

                painter.save()
                painter.setOpacity(0.82)
                painter.drawImage(overview, qimage)
                painter.setOpacity(1.0)
                painter.setPen(QPen(QColor(255, 255, 255, 220), 1))
                painter.drawRect(overview)
                painter.setPen(QPen(QColor("#f7f871"), 2))
                painter.drawRect(view_rect)
                painter.restore()

            def _zoom_viewport(self) -> tuple[float, float, float, float]:
                if self.rgb is None:
                    return 0.0, 0.0, 1.0, 1.0
                h, w, _ = self.rgb.shape
                zoom = max(1.0, min(float(self.zoom_factor), self.max_zoom_factor))
                self.zoom_factor = zoom
                view_h = max(float(h) / zoom, 1.0)
                view_w = max(float(w) / zoom, 1.0)
                if self.zoom_center is None:
                    center_row = (float(h) - 1.0) / 2.0
                    center_col = (float(w) - 1.0) / 2.0
                else:
                    center_row, center_col = self.zoom_center
                row0 = min(max(center_row - view_h / 2.0, 0.0), max(float(h) - view_h, 0.0))
                col0 = min(max(center_col - view_w / 2.0, 0.0), max(float(w) - view_w, 0.0))
                row1 = row0 + view_h
                col1 = col0 + view_w
                self.zoom_center = ((row0 + row1) / 2.0, (col0 + col1) / 2.0)
                return row0, col0, row1, col1

            def _clamp_zoom_center(self) -> None:
                if self.rgb is None:
                    self.zoom_center = None
                    return
                if self.zoom_factor <= 1.0:
                    self.zoom_factor = 1.0
                    self.zoom_center = None
                    return
                self._zoom_viewport()

            def zoom_by(self, factor: float, center: tuple[float, float] | None = None) -> None:
                if self.rgb is None:
                    return
                if center is None and self.zoom_center is None:
                    h, w, _ = self.rgb.shape
                    center = ((float(h) - 1.0) / 2.0, (float(w) - 1.0) / 2.0)
                if center is not None:
                    self.zoom_center = self._clamp_image_coords(center[0], center[1])
                self.zoom_factor = max(1.0, min(float(self.zoom_factor) * float(factor), self.max_zoom_factor))
                if self.zoom_factor <= 1.0001:
                    self.reset_zoom(update=False)
                else:
                    self._clamp_zoom_center()
                self.update()

            def reset_zoom(self, update: bool = True) -> None:
                self.zoom_factor = 1.0
                self.zoom_center = None
                self.drag_overview = False
                self.drag_pan = False
                self.pan_previous_pos = None
                if update:
                    self.update()

            def zoom_status_text(self) -> str:
                if self.zoom_factor <= 1.0:
                    return ""
                return f" | zoom {self.zoom_factor:.2f}x"

            def _image_to_widget(self, row: float, col: float) -> tuple[float, float]:
                if self.image_rect is None or self.viewport_image_rect is None:
                    return float(col), float(row)
                row0, col0, row1, col1 = self.viewport_image_rect
                x = float(self.image_rect.left()) + (float(col) - col0) / max(col1 - col0, 1.0e-6) * float(self.image_rect.width())
                y = float(self.image_rect.top()) + (float(row) - row0) / max(row1 - row0, 1.0e-6) * float(self.image_rect.height())
                return x, y

            def _current_image_scale(self) -> float:
                if self.image_rect is None or self.viewport_image_rect is None:
                    return 1.0
                row0, col0, row1, col1 = self.viewport_image_rect
                scale_x = float(self.image_rect.width()) / max(col1 - col0, 1.0e-6)
                scale_y = float(self.image_rect.height()) / max(row1 - row0, 1.0e-6)
                return min(scale_x, scale_y)

            def _handle_overview_press(self, pos: Any) -> bool:
                if self.zoom_factor <= 1.0 or self.overview_rect is None:
                    return False
                if not self.overview_rect.contains(pos):
                    return False
                self.drag_overview = True
                self._set_zoom_center_from_overview(pos)
                return True

            def _set_zoom_center_from_overview(self, pos: Any) -> None:
                if self.rgb is None or self.overview_rect is None:
                    return
                h, w, _ = self.rgb.shape
                col = (float(pos.x()) - float(self.overview_rect.left())) / max(float(self.overview_rect.width()), 1.0e-6) * float(w)
                row = (float(pos.y()) - float(self.overview_rect.top())) / max(float(self.overview_rect.height()), 1.0e-6) * float(h)
                self.zoom_center = self._clamp_image_coords(row, col)
                self._clamp_zoom_center()
                self.update()

            def _handle_pan_move(self, pos: Any) -> None:
                if self.pan_previous_pos is None or self.image_rect is None or self.viewport_image_rect is None:
                    return
                previous_x, previous_y = self.pan_previous_pos
                dx = float(pos.x()) - previous_x
                dy = float(pos.y()) - previous_y
                self.pan_previous_pos = (float(pos.x()), float(pos.y()))
                row0, col0, row1, col1 = self.viewport_image_rect
                center_row = (row0 + row1) / 2.0 - dy / max(float(self.image_rect.height()), 1.0e-6) * (row1 - row0)
                center_col = (col0 + col1) / 2.0 - dx / max(float(self.image_rect.width()), 1.0e-6) * (col1 - col0)
                self.zoom_center = (center_row, center_col)
                self._clamp_zoom_center()
                self.update()

            def mousePressEvent(self, event: Any) -> None:
                self.activate_callback()
                left_button = Qt.MouseButton.LeftButton if hasattr(Qt, "MouseButton") else Qt.LeftButton
                right_button = Qt.MouseButton.RightButton if hasattr(Qt, "MouseButton") else Qt.RightButton
                middle_button = Qt.MouseButton.MiddleButton if hasattr(Qt, "MouseButton") else Qt.MiddleButton
                if event.button() not in (left_button, right_button, middle_button):
                    return
                pos = self._event_pos(event)
                if event.button() == left_button and self._handle_overview_press(pos):
                    return
                if event.button() == middle_button and self.zoom_factor > 1.0 and self.image_rect is not None and self.image_rect.contains(pos):
                    self.drag_pan = True
                    self.pan_previous_pos = (float(pos.x()), float(pos.y()))
                    return
                coords = self._event_image_coords(event)
                if coords is None:
                    return
                row, col = coords
                tool = self.guide_tool_callback()
                if tool != LINE_TOOL_POINT:
                    self._handle_line_press(row, col, event.button() == right_button)
                    return
                nearest = self._nearest_editable_point(row, col)
                if event.button() == right_button:
                    if nearest is not None:
                        self.point_delete_callback(nearest)
                    return
                if nearest is not None:
                    self.drag_point = nearest
                    self.drag_offset = (nearest.row - row, nearest.col - col)
                    return
                self.callback(row, col)

            def mouseMoveEvent(self, event: Any) -> None:
                if self.drag_overview:
                    self._set_zoom_center_from_overview(self._event_pos(event))
                    return
                if self.drag_pan:
                    self._handle_pan_move(self._event_pos(event))
                    return
                if self.active_line is not None:
                    coords = self._event_image_coords(event, clamp=True)
                    if coords is None:
                        return
                    self._handle_line_move(*coords)
                    return
                if self.drag_point is None:
                    return
                coords = self._event_image_coords(event, clamp=True)
                if coords is None:
                    return
                row, col = coords
                row, col = self._clamp_image_coords(row + self.drag_offset[0], col + self.drag_offset[1])
                self.drag_point.row = row
                self.drag_point.col = col
                self.update()
                self.point_drag_callback(self.drag_point, row, col, False)

            def mouseReleaseEvent(self, event: Any) -> None:
                if self.drag_overview:
                    self.drag_overview = False
                    return
                if self.drag_pan:
                    self.drag_pan = False
                    self.pan_previous_pos = None
                    return
                if self.active_line is not None:
                    coords = self._event_image_coords(event, clamp=True)
                    if coords is not None:
                        self._handle_line_move(*coords)
                    self._finish_line_interaction()
                    return
                if self.drag_point is None:
                    return
                coords = self._event_image_coords(event, clamp=True)
                point = self.drag_point
                self.drag_point = None
                if coords is None:
                    return
                row, col = coords
                row, col = self._clamp_image_coords(row + self.drag_offset[0], col + self.drag_offset[1])
                point.row = row
                point.col = col
                self.point_drag_callback(point, row, col, True)

            def wheelEvent(self, event: Any) -> None:
                self.activate_callback()
                delta = event.angleDelta().y()
                ctrl = Qt.KeyboardModifier.ControlModifier if hasattr(Qt, "KeyboardModifier") else Qt.ControlModifier
                if event.modifiers() & ctrl:
                    coords = self._event_image_coords(event, clamp=True)
                    if coords is not None:
                        self.zoom_by(1.25 if delta > 0 else 0.8, center=coords)
                        event.accept()
                    return
                if delta > 0:
                    self.wheel_callback(-1)
                elif delta < 0:
                    self.wheel_callback(1)

            def update_cursor(self) -> None:
                tool = self.guide_tool_callback()
                if tool == LINE_TOOL_DRAW:
                    cursor = Qt.CursorShape.CrossCursor if hasattr(Qt, "CursorShape") else Qt.CrossCursor
                elif tool == LINE_TOOL_EDIT:
                    cursor = Qt.CursorShape.OpenHandCursor if hasattr(Qt, "CursorShape") else Qt.OpenHandCursor
                elif tool == LINE_TOOL_ERASE:
                    cursor = Qt.CursorShape.ForbiddenCursor if hasattr(Qt, "CursorShape") else Qt.ForbiddenCursor
                else:
                    cursor = Qt.CursorShape.ArrowCursor if hasattr(Qt, "CursorShape") else Qt.ArrowCursor
                self.setCursor(cursor)

            def enterEvent(self, event: Any) -> None:
                self.activate_callback()
                super().enterEvent(event)

            def _event_pos(self, event: Any) -> Any:
                return event.position() if hasattr(event, "position") else event.pos()

            def _event_image_coords(self, event: Any, clamp: bool = False) -> tuple[float, float] | None:
                if self.rgb is None or self.image_rect is None or self.viewport_image_rect is None:
                    return None
                pos = self._event_pos(event)
                if not clamp and not self.image_rect.contains(pos):
                    return None
                row0, col0, row1, col1 = self.viewport_image_rect
                col = col0 + (float(pos.x()) - float(self.image_rect.left())) / float(self.image_rect.width()) * (col1 - col0)
                row = row0 + (float(pos.y()) - float(self.image_rect.top())) / float(self.image_rect.height()) * (row1 - row0)
                if clamp:
                    row, col = self._clamp_image_coords(row, col)
                return row, col

            def _clamp_image_coords(self, row: float, col: float) -> tuple[float, float]:
                if self.rgb is None:
                    return row, col
                h, w, _ = self.rgb.shape
                return min(max(row, 0.0), float(h - 1)), min(max(col, 0.0), float(w - 1))

            def _nearest_editable_point(self, row: float, col: float) -> DisplayPoint | None:
                if self.rgb is None or self.image_rect is None:
                    return None
                scale = self._current_image_scale()
                hit_radius = 12.0 / max(scale, 1.0e-6)
                candidates = [
                    (float(np.hypot(point.row - row, point.col - col)), point)
                    for point in self.points
                    if point.editable
                ]
                if not candidates:
                    return None
                distance, point = min(candidates, key=lambda item: item[0])
                return point if distance <= hit_radius else None

            def _handle_line_press(self, row: float, col: float, right_click: bool) -> None:
                tool = self.guide_tool_callback()
                if right_click or tool == LINE_TOOL_ERASE:
                    line, _part = self._nearest_line(row, col)
                    if line is not None:
                        self.guide_deleted_callback(line)
                        self.update()
                    return
                if tool == LINE_TOOL_DRAW:
                    self.line_anchor = (row, col)
                    self.active_line = self._make_line(row, col, row, col)
                    self.active_line_part = "draw"
                    self.preview_line = self.active_line
                    self.update()
                    return
                if tool == LINE_TOOL_EDIT:
                    line, part = self._nearest_line(row, col)
                    if line is not None and part is not None:
                        self.active_line = line
                        self.active_line_part = part
                        self.line_drag_previous = (row, col)
                        if part == "start":
                            r1, c1, r2, c2 = self._line_to_image(line)
                            self.line_anchor = (r2, c2)
                        elif part == "end":
                            r1, c1, r2, c2 = self._line_to_image(line)
                            self.line_anchor = (r1, c1)
                        self.update()

            def _handle_line_move(self, row: float, col: float) -> None:
                if self.active_line is None or self.active_line_part is None:
                    return
                if self.active_line_part == "draw" and self.line_anchor is not None:
                    row, col = self._constrain_line_end(self.line_anchor, row, col)
                    self._set_line_image_points(self.active_line, self.line_anchor[0], self.line_anchor[1], row, col)
                elif self.active_line_part in ("start", "end") and self.line_anchor is not None:
                    row, col = self._constrain_line_end(self.line_anchor, row, col)
                    r1, c1, r2, c2 = self._line_to_image(self.active_line)
                    if self.active_line_part == "start":
                        self._set_line_image_points(self.active_line, row, col, r2, c2)
                    else:
                        self._set_line_image_points(self.active_line, r1, c1, row, col)
                elif self.active_line_part == "body" and self.line_drag_previous is not None:
                    prev_row, prev_col = self.line_drag_previous
                    delta_row = row - prev_row
                    delta_col = col - prev_col
                    r1, c1, r2, c2 = self._line_to_image(self.active_line)
                    self._set_line_image_points(self.active_line, r1 + delta_row, c1 + delta_col, r2 + delta_row, c2 + delta_col)
                    self.line_drag_previous = (row, col)
                self.update()

            def _finish_line_interaction(self) -> None:
                if self.active_line_part == "draw" and self.preview_line is not None:
                    r1, c1, r2, c2 = self._line_to_image(self.preview_line)
                    if float(np.hypot(r2 - r1, c2 - c1)) >= 3.0:
                        self.guide_created_callback(self.preview_line)
                self.preview_line = None
                self.active_line = None
                self.active_line_part = None
                self.line_anchor = None
                self.line_drag_previous = None
                self.update()

            def _paint_line(self, painter: Any, line: DisplayLine, selected: bool) -> None:
                from qtpy.QtGui import QColor, QPen

                r1, c1, r2, c2 = self._line_to_image(line)
                painter.setPen(QPen(QColor("#f7f871" if selected else line.color), 2))
                x1, y1 = self._image_to_widget(r1, c1)
                x2, y2 = self._image_to_widget(r2, c2)
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
                painter.setPen(QPen(QColor("#ffffff"), 1))
                for row, col in [(r1, c1), (r2, c2)]:
                    x, y = self._image_to_widget(row, col)
                    x = int(x)
                    y = int(y)
                    painter.drawRect(x - 3, y - 3, 6, 6)

            def _make_line(self, row1: float, col1: float, row2: float, col2: float) -> DisplayLine:
                line = DisplayLine(0.0, 0.0, 0.0, 0.0)
                self._set_line_image_points(line, row1, col1, row2, col2)
                return line

            def _line_to_image(self, line: DisplayLine) -> tuple[float, float, float, float]:
                if self.rgb is None:
                    return 0.0, 0.0, 0.0, 0.0
                h, w, _ = self.rgb.shape
                return (
                    line.row1_norm * float(max(h - 1, 1)),
                    line.col1_norm * float(max(w - 1, 1)),
                    line.row2_norm * float(max(h - 1, 1)),
                    line.col2_norm * float(max(w - 1, 1)),
                )

            def _set_line_image_points(self, line: DisplayLine, row1: float, col1: float, row2: float, col2: float) -> None:
                if self.rgb is None:
                    return
                h, w, _ = self.rgb.shape
                row1, col1 = self._clamp_image_coords(row1, col1)
                row2, col2 = self._clamp_image_coords(row2, col2)
                line.row1_norm = row1 / float(max(h - 1, 1))
                line.col1_norm = col1 / float(max(w - 1, 1))
                line.row2_norm = row2 / float(max(h - 1, 1))
                line.col2_norm = col2 / float(max(w - 1, 1))

            def _nearest_line(self, row: float, col: float) -> tuple[DisplayLine | None, str | None]:
                if self.rgb is None or self.image_rect is None:
                    return None, None
                scale = self._current_image_scale()
                hit_radius = 10.0 / max(scale, 1.0e-6)
                best: tuple[float, DisplayLine, str] | None = None
                for line in self.lines:
                    r1, c1, r2, c2 = self._line_to_image(line)
                    candidates = [
                        (float(np.hypot(row - r1, col - c1)), line, "start"),
                        (float(np.hypot(row - r2, col - c2)), line, "end"),
                        (self._distance_to_segment(row, col, r1, c1, r2, c2), line, "body"),
                    ]
                    nearest = min(candidates, key=lambda item: item[0])
                    if best is None or nearest[0] < best[0]:
                        best = nearest
                if best is None or best[0] > hit_radius:
                    return None, None
                return best[1], best[2]

            @staticmethod
            def _distance_to_segment(row: float, col: float, r1: float, c1: float, r2: float, c2: float) -> float:
                dr = r2 - r1
                dc = c2 - c1
                denom = dr * dr + dc * dc
                if denom <= 1.0e-8:
                    return float(np.hypot(row - r1, col - c1))
                t = max(0.0, min(1.0, ((row - r1) * dr + (col - c1) * dc) / denom))
                proj_row = r1 + t * dr
                proj_col = c1 + t * dc
                return float(np.hypot(row - proj_row, col - proj_col))

            def _constrain_line_end(self, anchor: tuple[float, float], row: float, col: float) -> tuple[float, float]:
                anchor_row, anchor_col = anchor
                constraint = self.guide_constraint_callback()
                if constraint == LINE_CONSTRAINT_HORIZONTAL:
                    return anchor_row, col
                if constraint == LINE_CONSTRAINT_VERTICAL:
                    return row, anchor_col
                if constraint == LINE_CONSTRAINT_ANGLE:
                    theta = np.deg2rad(float(self.guide_angle_callback()))
                    unit_row = float(np.sin(theta))
                    unit_col = float(np.cos(theta))
                    length = (row - anchor_row) * unit_row + (col - anchor_col) * unit_col
                    return anchor_row + length * unit_row, anchor_col + length * unit_col
                return row, col

        self.widget = _Canvas(
            click_callback,
            drag_callback,
            delete_callback,
            step_callback,
            line_tool_callback,
            line_constraint_callback,
            line_angle_callback,
            line_created_callback,
            line_deleted_callback,
            activate_callback,
        )
        self.widget.setParent(parent)


class DualImagePanel:
    def __init__(self, app: "DualViewWorkbench", title: str, source: str, view: str):
        from qtpy.QtWidgets import QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget
        from qtpy.QtCore import Qt

        self.app = app
        self.widget = QWidget()
        self.guide_lines: list[DisplayLine] = []
        self.title_label = QLabel(title)
        self.source_combo = QComboBox()
        self.source_combo.addItems([SOURCE_CT, SOURCE_MRI, SOURCE_MRI_ON_CT])
        self.source_combo.setCurrentText(source)
        self.view_combo = QComboBox()
        self.view_combo.addItems([VIEW_AXIAL, VIEW_CORONAL, VIEW_SAGITTAL])
        self.view_combo.setCurrentText(view)
        self.window_combo = QComboBox()
        self.window_combo.addItems(["Soft", "Bone"])
        self.line_tool_combo = QComboBox()
        self.line_tool_combo.addItems([LINE_TOOL_POINT, LINE_TOOL_DRAW, LINE_TOOL_EDIT, LINE_TOOL_ERASE])
        self.line_constraint_combo = QComboBox()
        self.line_constraint_combo.addItems(
            [LINE_CONSTRAINT_FREE, LINE_CONSTRAINT_HORIZONTAL, LINE_CONSTRAINT_VERTICAL, LINE_CONSTRAINT_ANGLE]
        )
        self.line_angle_spin = QDoubleSpinBox()
        self.line_angle_spin.setRange(-180.0, 180.0)
        self.line_angle_spin.setDecimals(1)
        self.line_angle_spin.setSingleStep(5.0)
        self.line_angle_spin.setValue(45.0)
        clear_lines_button = QPushButton("Clear lines")
        self.canvas = SliceCanvas(
            self.widget,
            self._on_click,
            self._on_drag_point,
            self._on_delete_point,
            self.step_slice,
            self.line_tool,
            self.line_constraint,
            self.line_angle,
            self._on_line_created,
            self._on_line_deleted,
            self._on_activate,
        ).widget
        self.slider = QSlider(Qt.Orientation.Horizontal if hasattr(Qt, "Orientation") else Qt.Horizontal)
        self.slice_label = QLabel("slice 0/0")

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Source"))
        controls.addWidget(self.source_combo)
        controls.addWidget(QLabel("View"))
        controls.addWidget(self.view_combo)
        controls.addWidget(QLabel("Window"))
        controls.addWidget(self.window_combo)

        line_controls = QHBoxLayout()
        line_controls.addWidget(QLabel("Guide"))
        line_controls.addWidget(self.line_tool_combo)
        line_controls.addWidget(QLabel("Constraint"))
        line_controls.addWidget(self.line_constraint_combo)
        line_controls.addWidget(QLabel("Angle"))
        line_controls.addWidget(self.line_angle_spin)
        line_controls.addWidget(clear_lines_button)

        layout = QVBoxLayout(self.widget)
        layout.addWidget(self.title_label)
        layout.addLayout(controls)
        layout.addLayout(line_controls)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.slider)
        layout.addWidget(self.slice_label)

        self.source_combo.currentTextChanged.connect(lambda _: self.app.update_panel_range(self))
        self.view_combo.currentTextChanged.connect(lambda _: self.app.update_panel_range(self))
        self.window_combo.currentTextChanged.connect(lambda _: self.app.render_panel(self))
        self.slider.valueChanged.connect(lambda _: self.app.render_panel(self))
        self.line_tool_combo.currentTextChanged.connect(lambda _: self.canvas.update_cursor())
        clear_lines_button.clicked.connect(self.clear_lines)

    def source(self) -> str:
        return str(self.source_combo.currentText())

    def view(self) -> str:
        return str(self.view_combo.currentText())

    def window(self) -> str:
        return str(self.window_combo.currentText())

    def slice_index(self) -> int:
        return int(self.slider.value())

    def set_source(self, source: str) -> None:
        self.source_combo.setCurrentText(source)

    def set_view(self, view: str) -> None:
        self.view_combo.setCurrentText(view)

    def step_slice(self, delta: int) -> None:
        self.slider.setValue(max(self.slider.minimum(), min(self.slider.maximum(), self.slider.value() + int(delta))))

    def line_tool(self) -> str:
        return str(self.line_tool_combo.currentText())

    def line_constraint(self) -> str:
        return str(self.line_constraint_combo.currentText())

    def line_angle(self) -> float:
        return float(self.line_angle_spin.value())

    def clear_lines(self) -> None:
        self.guide_lines.clear()
        self.app.render_panel(self)

    def _on_click(self, row: float, col: float) -> None:
        self._on_activate()
        self.app.save_click(self, row, col)

    def _on_drag_point(self, point: DisplayPoint, row: float, col: float, final: bool) -> None:
        self._on_activate()
        self.app.move_display_point(self, point, row, col, final)

    def _on_delete_point(self, point: DisplayPoint) -> None:
        self._on_activate()
        self.app.delete_display_point(self, point)

    def _on_line_created(self, line: DisplayLine) -> None:
        if line not in self.guide_lines:
            self.guide_lines.append(line)
        self.app.render_panel(self)

    def _on_line_deleted(self, line: DisplayLine) -> None:
        if line in self.guide_lines:
            self.guide_lines.remove(line)
        self.app.render_panel(self)

    def _on_activate(self) -> None:
        self.app.set_active_zoom_panel(self)

    def zoom_by(self, factor: float) -> None:
        self.canvas.zoom_by(factor)
        self.app.render_panel(self)

    def reset_zoom(self) -> None:
        self.canvas.reset_zoom()
        self.app.render_panel(self)


class DualViewWorkbench:
    def __init__(
        self,
        manifest_csv: str | Path,
        patient_id: str,
        db_path: str | Path,
        ct_series_description: str = "AX",
        mri_series_description: str = DEFAULT_MRI_SERIES,
        work_queue_csv: str | Path | None = None,
        annotator_id: str = "default",
    ):
        self.manifest_csv = Path(manifest_csv)
        self.manifest_rows = read_manifest_rows(self.manifest_csv)
        self.patient_id = str(patient_id)
        self.db_path = Path(db_path)
        self.work_queue_csv = Path(work_queue_csv) if work_queue_csv else None
        self.queue_rows = self._read_work_queue(self.work_queue_csv)
        self.queue_index = self._queue_index_for_patient(self.patient_id)
        self.ct_series_description = ct_series_description
        self.requested_mri_series_description = mri_series_description
        self.mri_series_description = mri_series_description
        self.annotator_id = annotator_id
        self.store = AnnotationStore(self.db_path)
        self.current_label = LANDMARK_LABELS[0]
        self.current_visibility = "visible"
        self.current_quality = 0
        self._mri_candidate_cache: dict[str, dict[str, dict[str, str]]] = {}

        self.window = None
        self.patient_edit = None
        self.mri_combo = None
        self.label_combo = None
        self.globe_side_combo = None
        self.center_override_combo = None
        self.visibility_combo = None
        self.quality_combo = None
        self.pitch_spin = None
        self.scale_x_spin = None
        self.scale_y_spin = None
        self.scale_z_spin = None
        self.progress_label = None
        self.status_label = None
        self.left_panel: DualImagePanel | None = None
        self.right_panel: DualImagePanel | None = None
        self.active_zoom_panel: DualImagePanel | None = None
        self.three_d_panel: Globe3DPanel | None = None
        self.shift_shortcut_filter = None
        self.live_overlay_timer = None
        self.live_overlay_dirty = False
        self.live_globe_timer = None
        self.live_globe_dirty = False
        self.live_overlay_interval_ms = 150
        self.loading_manual_parameters = False
        self.autosave_disabled_for_patient_id: str | None = None
        self.center_override_click_target: tuple[str, str] | None = None

        self.ct_volume = None
        self.mri_volume = None
        self.ct_lps = None
        self.mri_lps = None
        self.ct_soft_lps = None
        self.ct_bone_lps = None
        self.mri_display_lps = None
        self.overlay_result = None
        self.overlay_ct_soft_lps = None
        self.overlay_ct_bone_lps = None
        self.globe_registration_result = None
        self.globe_mri_on_ct_lps = None
        self.globe_mri_on_ct_display_lps = None

    def launch(self) -> None:
        from qtpy.QtGui import QKeySequence, QShortcut
        from qtpy.QtWidgets import (
            QApplication,
            QComboBox,
            QDoubleSpinBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QPushButton,
            QSplitter,
            QVBoxLayout,
            QWidget,
        )
        from qtpy.QtCore import QEvent, QObject, QTimer, Qt

        class _ShiftShortcutFilter(QObject):
            def __init__(self, callback: Any):
                super().__init__()
                self.callback = callback

            def eventFilter(self, obj: Any, event: Any) -> bool:
                key_press = QEvent.Type.KeyPress if hasattr(QEvent, "Type") else QEvent.KeyPress
                shift_key = Qt.Key.Key_Shift if hasattr(Qt, "Key") else Qt.Key_Shift
                if event.type() == key_press and event.key() == shift_key and not event.isAutoRepeat():
                    self.callback()
                    event.accept()
                    return True
                return False

        app = QApplication.instance() or QApplication([])
        self.window = QMainWindow()
        self.window.setWindowTitle(f"Dual CT/MRI landmark workbench - {self.patient_id}")

        root = QWidget()
        root_layout = QVBoxLayout(root)
        top = QHBoxLayout()
        self.patient_edit = QLineEdit(self.patient_id)
        load_button = QPushButton("Load patient")
        prev_button = QPushButton("Previous")
        next_button = QPushButton("Next")
        delete_patient_button = QPushButton("Delete current data")
        self.progress_label = QLabel()
        self.mri_combo = QComboBox()
        self.mri_combo.addItems(list(MRI_CANDIDATE_SERIES))
        self.mri_combo.setCurrentText(self.requested_mri_series_description)
        load_mri_button = QPushButton("Load MRI")
        self.globe_side_combo = QComboBox()
        self.globe_side_combo.addItems(["L", "R"])
        left_eye_button = QPushButton("Manual L (Q)")
        right_eye_button = QPushButton("Manual R (W)")
        globe_compute_button = QPushButton("Compute globe MRI-on-CT")
        save_globe_button = QPushButton("Save globe transform")
        self.center_override_combo = QComboBox()
        self.center_override_combo.addItems(["CT LC", "CT RC", "MRI LC", "MRI RC"])
        set_center_button = QPushButton("Set center by click")
        clear_center_button = QPushButton("Clear forced center")
        self.pitch_spin = self._make_spin(-45.0, 45.0, 0.0, 1.0)
        self.scale_x_spin = self._make_spin(0.5, 1.5, 1.0, 0.01)
        self.scale_y_spin = self._make_spin(0.5, 1.5, 1.0, 0.01)
        self.scale_z_spin = self._make_spin(0.5, 1.5, 1.0, 0.01)

        for widget in [
            QLabel("Patient"),
            self.patient_edit,
            load_button,
            prev_button,
            next_button,
            self.progress_label,
            delete_patient_button,
            QLabel("MRI"),
            self.mri_combo,
            load_mri_button,
            QLabel("Manual eye"),
            self.globe_side_combo,
            left_eye_button,
            right_eye_button,
        ]:
            top.addWidget(widget)
        root_layout.addLayout(top)

        manual = QHBoxLayout()
        for widget in [
            globe_compute_button,
            save_globe_button,
            QLabel("Center"),
            self.center_override_combo,
            set_center_button,
            clear_center_button,
            QLabel("Pitch"),
            self.pitch_spin,
            QLabel("Scale X(LR)"),
            self.scale_x_spin,
            QLabel("Scale Y(AP)"),
            self.scale_y_spin,
            QLabel("Scale Z(SI)"),
            self.scale_z_spin,
        ]:
            manual.addWidget(widget)
        root_layout.addLayout(manual)

        splitter = QSplitter(Qt.Orientation.Horizontal if hasattr(Qt, "Orientation") else Qt.Horizontal)
        self.left_panel = DualImagePanel(self, "Left view", SOURCE_CT, VIEW_AXIAL)
        self.right_panel = DualImagePanel(self, "Right view", SOURCE_MRI, VIEW_CORONAL)
        self.active_zoom_panel = self.right_panel
        self.three_d_panel = Globe3DPanel(self)
        splitter.addWidget(self.left_panel.widget)
        splitter.addWidget(self.right_panel.widget)
        splitter.addWidget(self.three_d_panel.widget)
        splitter.setSizes([760, 760, 760])
        root_layout.addWidget(splitter, stretch=1)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        root_layout.addWidget(self.status_label)
        self.window.setCentralWidget(root)

        load_button.clicked.connect(lambda: self.load_patient(str(self.patient_edit.text())))
        prev_button.clicked.connect(self.previous_patient)
        next_button.clicked.connect(self.next_patient)
        delete_patient_button.clicked.connect(lambda: self.delete_current_patient_data(confirm=True))
        load_mri_button.clicked.connect(lambda: self.load_mri_series(str(self.mri_combo.currentText())))
        self.globe_side_combo.currentTextChanged.connect(lambda value: self.set_globe_side(str(value), echo=False))
        left_eye_button.clicked.connect(lambda: self.set_globe_side("L"))
        right_eye_button.clicked.connect(lambda: self.set_globe_side("R"))
        globe_compute_button.clicked.connect(lambda: self.update_globe_registration_preview(write_outputs=False, switch_to_preview=True, echo=True))
        save_globe_button.clicked.connect(lambda: self.update_globe_registration_preview(write_outputs=True, switch_to_preview=True, echo=True))
        set_center_button.clicked.connect(self.start_center_override_click)
        clear_center_button.clicked.connect(self.clear_selected_center_override)
        for spin in [self.pitch_spin, self.scale_x_spin, self.scale_y_spin, self.scale_z_spin]:
            spin.valueChanged.connect(lambda _value: self.on_manual_parameter_changed())

        QShortcut(QKeySequence("q"), self.window).activated.connect(lambda: self.set_globe_side("L"))
        QShortcut(QKeySequence("w"), self.window).activated.connect(lambda: self.set_globe_side("R"))
        QShortcut(QKeySequence("n"), self.window).activated.connect(self.next_patient)
        QShortcut(QKeySequence("p"), self.window).activated.connect(self.previous_patient)
        QShortcut(QKeySequence("v"), self.window).activated.connect(lambda: self.set_3d_mode(MODE_VIEW))
        QShortcut(QKeySequence("t"), self.window).activated.connect(lambda: self.set_3d_mode(MODE_PITCH))
        QShortcut(QKeySequence("x"), self.window).activated.connect(lambda: self.set_3d_mode(MODE_SCALE_X))
        QShortcut(QKeySequence("y"), self.window).activated.connect(lambda: self.set_3d_mode(MODE_SCALE_Y))
        QShortcut(QKeySequence("z"), self.window).activated.connect(lambda: self.set_3d_mode(MODE_SCALE_Z))
        for sequence in ["Ctrl++", "Ctrl+=", "Meta++", "Meta+="]:
            QShortcut(QKeySequence(sequence), self.window).activated.connect(lambda factor=1.25: self.zoom_active_panel(factor))
        for sequence in ["Ctrl+-", "Meta+-"]:
            QShortcut(QKeySequence(sequence), self.window).activated.connect(lambda factor=0.8: self.zoom_active_panel(factor))
        for sequence in ["Ctrl+0", "Meta+0"]:
            QShortcut(QKeySequence(sequence), self.window).activated.connect(self.reset_active_panel_zoom)
        QShortcut(QKeySequence("Esc"), self.window).activated.connect(self.cancel_center_override_click)
        self.shift_shortcut_filter = _ShiftShortcutFilter(
            lambda: self.update_globe_registration_preview(write_outputs=False, switch_to_preview=True, echo=True)
        )
        app.installEventFilter(self.shift_shortcut_filter)
        self.live_overlay_timer = QTimer(self.window)
        self.live_overlay_timer.setSingleShot(True)
        self.live_overlay_timer.timeout.connect(self._on_live_overlay_timer)
        self.live_globe_timer = QTimer(self.window)
        self.live_globe_timer.setSingleShot(True)
        self.live_globe_timer.timeout.connect(self._on_live_globe_timer)

        self.load_patient(self.patient_id)
        self.window.resize(2400, 1050)
        self.window.show()
        app.exec()

    @staticmethod
    def _make_spin(minimum: float, maximum: float, value: float, step: float) -> Any:
        from qtpy.QtWidgets import QDoubleSpinBox

        spin = QDoubleSpinBox()
        spin.setRange(float(minimum), float(maximum))
        spin.setDecimals(3 if step < 0.1 else 1)
        spin.setSingleStep(float(step))
        spin.setValue(float(value))
        return spin

    def load_patient(self, patient_id: str) -> None:
        patient_id = str(patient_id).strip()
        if not patient_id:
            self.set_status("Patient id is empty.")
            return
        self.autosave_current_patient_before_switch()
        queue_row = self._queue_row_for_patient(patient_id)
        try:
            ct_volume = self._load_ct_volume(patient_id, queue_row)
            mri_volume = self._load_mri_volume(patient_id, self.requested_mri_series_description, queue_row)
        except Exception as exc:
            self.set_status(f"Failed to load patient {patient_id}: {exc}")
            return
        self.patient_id = patient_id
        self.queue_index = self._queue_index_for_patient(patient_id)
        self.ct_volume = ct_volume
        self.mri_volume = mri_volume
        self.ct_series_description = ct_volume.selection.series_description
        self.mri_series_description = mri_volume.selection.series_description
        self.requested_mri_series_description = self.mri_series_description
        self.center_override_click_target = None
        self.overlay_result = None
        self.overlay_ct_soft_lps = None
        self.overlay_ct_bone_lps = None
        self.globe_registration_result = None
        self.globe_mri_on_ct_lps = None
        self.globe_mri_on_ct_display_lps = None
        self.prepare_display_volumes()
        self.reset_panel_sources_for_loaded_patient()
        if self.patient_edit is not None:
            self.patient_edit.setText(self.patient_id)
        if self.mri_combo is not None:
            self.mri_combo.setCurrentText(self.mri_series_description)
        if self.window is not None:
            self.window.setWindowTitle(f"Dual CT/MRI landmark workbench - {self.patient_id}")
        self.load_manual_parameters_for_current_series()
        self.refresh_all_panels()
        self.refresh_3d_panel()
        self.update_progress_label()
        self.set_status(self.status_text())

    def reset_panel_sources_for_loaded_patient(self) -> None:
        if self.left_panel is not None:
            self.left_panel.set_source(SOURCE_CT)
            self.left_panel.set_view(VIEW_AXIAL)
        if self.right_panel is not None:
            self.right_panel.set_source(SOURCE_MRI)
            self.right_panel.set_view(VIEW_CORONAL)

    def delete_current_patient_data(self, confirm: bool = True) -> None:
        patient_id = str(self.patient_id)
        counts = self.store.count_patient_data(patient_id)
        output_dir = self.patient_output_dir(patient_id)
        output_exists = output_dir.exists()
        total_db_rows = sum(counts.values())
        if confirm and self.window is not None:
            from qtpy.QtWidgets import QMessageBox

            yes_button = QMessageBox.StandardButton.Yes if hasattr(QMessageBox, "StandardButton") else QMessageBox.Yes
            no_button = QMessageBox.StandardButton.No if hasattr(QMessageBox, "StandardButton") else QMessageBox.No
            message = (
                f"Delete saved data for current patient {patient_id}?\n\n"
                f"DB rows: {total_db_rows}\n"
                f"- landmarks: {counts.get('landmarks', 0)}\n"
                f"- annotation_events: {counts.get('annotation_events', 0)}\n"
                f"- globe_surface_points: {counts.get('globe_surface_points', 0)}\n"
                f"- globe_manual_parameters: {counts.get('globe_manual_parameters', 0)}\n"
                f"- globe_center_overrides: {counts.get('globe_center_overrides', 0)}\n"
                f"Output folder: {'yes' if output_exists else 'no'}\n\n"
                "This only deletes the saved annotation/registration outputs, not the original DICOM data."
            )
            choice = QMessageBox.question(self.window, "Delete current patient data", message, yes_button | no_button, no_button)
            if choice != yes_button:
                self.set_status(f"Delete cancelled for patient {patient_id}.")
                return

        deleted_counts = self.store.delete_patient_data(patient_id)
        deleted_output = False
        if output_exists:
            resolved = output_dir.resolve()
            expected_parent = self.db_path.parent.resolve()
            if resolved.parent != expected_parent or resolved.name != patient_id:
                raise RuntimeError(f"Refusing to delete unexpected output path: {resolved}")
            shutil.rmtree(resolved)
            deleted_output = True

        self.autosave_disabled_for_patient_id = patient_id
        self.overlay_result = None
        self.overlay_ct_soft_lps = None
        self.overlay_ct_bone_lps = None
        self.globe_registration_result = None
        self.globe_mri_on_ct_lps = None
        self.globe_mri_on_ct_display_lps = None
        self.set_manual_parameter_spinboxes(0.0, (1.0, 1.0, 1.0))
        self.reset_panel_sources_for_loaded_patient()
        self.refresh_all_panels()
        self.refresh_3d_panel()
        self.set_status(
            f"Deleted saved data for patient {patient_id}.\n"
            f"DB rows removed: {sum(deleted_counts.values())}; output folder removed: {deleted_output}\n"
            f"{self.status_text()}"
        )

    def load_mri_series(self, series_description: str) -> None:
        self.requested_mri_series_description = str(series_description)
        if not self._mri_candidate_row(self.patient_id, self.requested_mri_series_description):
            self.set_status(f"Patient {self.patient_id} has no {self.requested_mri_series_description}.\n{self.mri_availability_text()}")
            return
        self.load_patient(self.patient_id)

    def previous_patient(self) -> None:
        self._neighbor_patient(forward=False)

    def next_patient(self) -> None:
        self._neighbor_patient(forward=True)

    def _neighbor_patient(self, forward: bool) -> None:
        if not self.queue_rows:
            self.set_status("No work queue is loaded.")
            return
        description = self.requested_mri_series_description
        indices = [
            index
            for index, row in enumerate(self.queue_rows)
            if self._mri_candidate_row(str(row.get("patient_id", "")), description)
        ]
        current = self.queue_index if self.queue_index is not None else (-1 if forward else len(self.queue_rows))
        candidates = [index for index in indices if index > current] if forward else [index for index in indices if index < current]
        if not candidates:
            self.set_status(f"No {'next' if forward else 'previous'} queued patient with {description}.")
            return
        next_index = candidates[0] if forward else candidates[-1]
        self.load_patient(str(self.queue_rows[next_index]["patient_id"]))

    def prepare_display_volumes(self) -> None:
        if self.ct_volume is None or self.mri_volume is None:
            return
        self.ct_lps = self.orient_to_lps(self.ct_volume.image)
        self.mri_lps = self.orient_to_lps(self.mri_volume.image)
        self.ct_soft_lps = ct_window(self.sitk_array(self.ct_lps), -100, 200)
        self.ct_bone_lps = ct_window(self.sitk_array(self.ct_lps), -500, 1500)
        self.mri_display_lps = normalize_percentile(self.sitk_array(self.mri_lps))

    def refresh_all_panels(self) -> None:
        for panel in [self.left_panel, self.right_panel]:
            if panel is not None:
                self.update_panel_range(panel)

    def update_panel_range(self, panel: DualImagePanel) -> None:
        array = self.array_for_panel(panel)
        if array is None:
            return
        axis = self.slice_axis(panel.view())
        max_index = max(int(array.shape[axis]) - 1, 0)
        old = panel.slider.value()
        panel.slider.blockSignals(True)
        panel.slider.setMinimum(0)
        panel.slider.setMaximum(max_index)
        panel.slider.setValue(min(old, max_index))
        panel.slider.blockSignals(False)
        self.render_panel(panel)

    def render_panel(self, panel: DualImagePanel) -> None:
        array = self.array_for_panel(panel)
        if array is None:
            return
        view = panel.view()
        source = panel.source()
        slice_index = min(panel.slice_index(), max(int(array.shape[self.slice_axis(view)]) - 1, 0))
        image = self.display_slice(panel, array, slice_index)
        if source == SOURCE_OVERLAY and self.overlay_ct_soft_lps is not None:
            overlay_array = self.overlay_ct_bone_lps if panel.window() == "Bone" and self.overlay_ct_bone_lps is not None else self.overlay_ct_soft_lps
            overlay = self.display_slice(panel, overlay_array, slice_index)
            rgb = self.overlay_rgb(image, overlay)
        elif source == SOURCE_MRI_ON_CT and self.globe_mri_on_ct_display_lps is not None:
            overlay = self.display_slice(panel, self.globe_mri_on_ct_display_lps, slice_index)
            rgb = self.overlay_rgb_color(image, overlay, color=(40.0, 220.0, 255.0))
        else:
            rgb = self.gray_rgb(image)
        points = self.display_points_for_panel(panel)
        panel.canvas.set_scene(rgb, points, panel.guide_lines, self.lr_corner_labels_for_panel(panel))
        panel.slice_label.setText(
            f"slice {slice_index} / {panel.slider.maximum()} | {panel.source()} {panel.view()}{panel.canvas.zoom_status_text()}"
        )

    def array_for_panel(self, panel: DualImagePanel) -> np.ndarray | None:
        source = panel.source()
        if source == SOURCE_CT:
            return self.ct_bone_lps if panel.window() == "Bone" else self.ct_soft_lps
        if source == SOURCE_MRI:
            return self.mri_display_lps
        if source == SOURCE_OVERLAY:
            if self.mri_display_lps is None:
                return None
            return self.mri_display_lps
        if source == SOURCE_MRI_ON_CT:
            return self.ct_bone_lps if panel.window() == "Bone" else self.ct_soft_lps
        return None

    def image_for_panel(self, panel: DualImagePanel) -> Any | None:
        source = panel.source()
        if source == SOURCE_CT:
            return self.ct_lps
        if source in (SOURCE_MRI, SOURCE_OVERLAY):
            return self.mri_lps
        if source == SOURCE_MRI_ON_CT:
            return self.ct_lps
        return None

    def save_click(self, panel: DualImagePanel, row: float, col: float) -> None:
        if self.center_override_click_target is not None:
            modality, side = self.center_override_click_target
            self.save_globe_center_override_click(panel, row, col, modality, side)
            return
        if self.click_target() == CLICK_TARGET_GLOBE:
            self.save_globe_surface_click(panel, row, col)
            return
        modality = "CT" if panel.source() == SOURCE_CT else "MRI"
        self.save_panel_point(
            panel=panel,
            row=row,
            col=col,
            modality=modality,
            label=self.current_label,
            event_type="upsert",
            refresh=True,
            echo=True,
        )

    def click_target(self) -> str:
        return CLICK_TARGET_GLOBE

    def globe_side(self) -> str:
        if self.globe_side_combo is None:
            return "L"
        return str(self.globe_side_combo.currentText()).upper()

    def selected_center_override_target(self) -> tuple[str, str]:
        text = str(self.center_override_combo.currentText() if self.center_override_combo is not None else "CT LC").upper()
        modality = "MRI" if text.startswith("MRI") else "CT"
        side = "R" if "RC" in text else "L"
        return modality, side

    def center_override_label(self, modality: str, side: str) -> str:
        return f"{modality} {side}C"

    def start_center_override_click(self) -> None:
        self.center_override_click_target = self.selected_center_override_target()
        modality, side = self.center_override_click_target
        self.set_status(
            f"Click a native {modality} view to force {self.center_override_label(modality, side)}.\n"
            f"Existing surface points stay unchanged.",
            echo=True,
        )

    def cancel_center_override_click(self) -> None:
        if self.center_override_click_target is None:
            return
        modality, side = self.center_override_click_target
        self.center_override_click_target = None
        self.set_status(
            f"Cancelled forced {self.center_override_label(modality, side)} click mode; normal globe surface point mode is active.\n"
            f"{self.status_text()}",
            echo=True,
        )

    def clear_selected_center_override(self) -> None:
        modality, side = self.selected_center_override_target()
        series_uid = self.series_uid_for_modality(modality)
        if not series_uid:
            self.set_status(f"Cannot clear {self.center_override_label(modality, side)}: no active {modality} series.")
            return
        deleted = self.store.delete_globe_center_override(self.patient_id, modality, series_uid, side)
        self.center_override_click_target = None
        self.autosave_disabled_for_patient_id = None
        self.schedule_live_globe_registration_update(final=True)
        self.refresh_all_panels()
        self.refresh_3d_panel()
        action = "Cleared" if deleted else "No forced center to clear for"
        self.set_status(f"{action} {self.center_override_label(modality, side)}.\n{self.status_text()}")

    def save_globe_center_override_click(
        self,
        panel: DualImagePanel,
        row: float,
        col: float,
        modality: str,
        side: str,
    ) -> None:
        record = self.globe_center_override_record_from_display(panel, row, col, modality, side)
        if record is None:
            return
        self.center_override_click_target = None
        self.autosave_disabled_for_patient_id = None
        self.store.upsert_globe_center_override(record)
        self.schedule_live_globe_registration_update(final=True)
        self.refresh_all_panels()
        self.refresh_3d_panel()
        lps_text = ", ".join(f"{value:.2f}" for value in record.physical_lps_mm)
        self.set_status(
            f"Forced {self.center_override_label(record.modality, record.side)} at LPS=({lps_text}).\n"
            f"{self.status_text()}"
        )

    def globe_center_override_record_from_display(
        self,
        panel: DualImagePanel,
        row: float,
        col: float,
        modality: str,
        side: str,
    ) -> GlobeCenterOverrideRecord | None:
        modality = str(modality).upper()
        side = str(side).upper()
        if panel.source() != (SOURCE_CT if modality == "CT" else SOURCE_MRI):
            self.set_status(f"Use a native {modality} view to set {self.center_override_label(modality, side)}.")
            return None
        volume = self.volume_for_modality(modality)
        panel_image = self.panel_image_for_modality(panel, modality)
        if volume is None or panel_image is None:
            return None
        index_xyz = self.display_to_index_xyz(panel, panel.slice_index(), row, col)
        try:
            physical_lps = tuple(float(v) for v in panel_image.TransformContinuousIndexToPhysicalPoint(index_xyz))
            native_index_xyz = tuple(float(v) for v in volume.image.TransformPhysicalPointToContinuousIndex(physical_lps))
        except Exception as exc:
            self.set_status(f"Could not map forced center to DICOM physical coordinate: {exc}")
            return None
        native_index_xyz = self.clamp_tiny_index_drift(native_index_xyz, volume.image.GetSize())
        physical_lps = tuple(float(v) for v in volume.image.TransformContinuousIndexToPhysicalPoint(native_index_xyz))
        return GlobeCenterOverrideRecord(
            patient_id=self.patient_id,
            study_uid=volume.selection.study_uid,
            series_uid=volume.selection.series_uid,
            modality=modality,
            side=side,
            physical_lps_mm=[float(v) for v in physical_lps],
            annotator_id=self.annotator_id,
        )

    def save_globe_surface_click(self, panel: DualImagePanel, row: float, col: float) -> None:
        auto_side = self.uses_auto_globe_side(panel)
        side = self.globe_side_for_display_point(panel, row, col)
        record = self.globe_surface_record_from_display(panel, row, col, side)
        if record is None:
            return
        if auto_side:
            self.set_globe_side(record.side, echo=False)
        self.autosave_disabled_for_patient_id = None
        point_id = self.store.add_globe_surface_point(record)
        self.schedule_live_globe_registration_update(final=True)
        self.refresh_all_panels()
        self.refresh_3d_panel()
        side_mode = "auto" if auto_side else "manual"
        self.set_status(f"Saved {record.modality} {record.side} globe surface point #{point_id} ({side_mode} side)\n{self.status_text()}")

    def uses_auto_globe_side(self, panel: DualImagePanel) -> bool:
        return panel.source() in (SOURCE_CT, SOURCE_MRI) and panel.view() in (VIEW_AXIAL, VIEW_CORONAL)

    def globe_side_for_display_point(self, panel: DualImagePanel, row: float, col: float) -> str:
        if not self.uses_auto_globe_side(panel):
            return self.globe_side()
        panel_image = self.image_for_panel(panel)
        if panel_image is None:
            return self.globe_side()
        try:
            index_xyz = self.display_to_index_xyz(panel, panel.slice_index(), row, col)
            physical_lps = panel_image.TransformContinuousIndexToPhysicalPoint(index_xyz)
            center_x = self.image_center_lps_x(panel_image)
        except Exception:
            return self.globe_side()
        return "L" if float(physical_lps[0]) >= center_x else "R"

    def lr_corner_labels_for_panel(self, panel: DualImagePanel) -> tuple[str, str] | None:
        if panel.view() not in (VIEW_AXIAL, VIEW_CORONAL):
            return None
        panel_image = self.image_for_panel(panel)
        raw_height = self.raw_slice_height(panel)
        raw_width = self.raw_slice_width(panel)
        if panel_image is None or raw_height <= 0 or raw_width <= 1:
            return None
        display_height = self.display_slice_height(panel, raw_height)
        display_width = self.display_slice_width(panel, raw_width)
        if display_width <= 1:
            return None
        sample_row = float(max(display_height - 1, 0)) / 2.0
        slice_index = panel.slice_index()
        try:
            left_index = self.display_to_index_xyz(panel, slice_index, sample_row, 0.0)
            right_index = self.display_to_index_xyz(panel, slice_index, sample_row, float(display_width - 1))
            left_x = float(panel_image.TransformContinuousIndexToPhysicalPoint(left_index)[0])
            right_x = float(panel_image.TransformContinuousIndexToPhysicalPoint(right_index)[0])
        except Exception:
            return None
        if abs(left_x - right_x) < 1e-6:
            return None
        return ("L", "R") if left_x > right_x else ("R", "L")

    def globe_surface_record_from_display(
        self,
        panel: DualImagePanel,
        row: float,
        col: float,
        side: str,
    ) -> GlobeSurfacePointRecord | None:
        if panel.source() == SOURCE_CT:
            modality = "CT"
            volume = self.ct_volume
            panel_image = self.ct_lps
        elif panel.source() == SOURCE_MRI:
            modality = "MRI"
            volume = self.mri_volume
            panel_image = self.mri_lps
        else:
            self.set_status("Globe surface points can be added only on native CT or native MRI views.")
            return None
        if volume is None or panel_image is None:
            return None
        index_xyz = self.display_to_index_xyz(panel, panel.slice_index(), row, col)
        try:
            physical_lps = tuple(float(v) for v in panel_image.TransformContinuousIndexToPhysicalPoint(index_xyz))
            native_index_xyz = tuple(float(v) for v in volume.image.TransformPhysicalPointToContinuousIndex(physical_lps))
        except Exception as exc:
            self.set_status(f"Could not map globe point to DICOM physical coordinate: {exc}")
            return None
        native_index_xyz = self.clamp_tiny_index_drift(native_index_xyz, volume.image.GetSize())
        physical_lps = tuple(float(v) for v in volume.image.TransformContinuousIndexToPhysicalPoint(native_index_xyz))
        native_zyx = itk_xyz_to_napari_zyx(native_index_xyz)
        if not point_inside_array_shape_zyx(native_zyx, volume.array_zyx.shape):
            self.set_status(f"Rejected {modality} globe point outside native volume: {native_zyx}")
            return None
        return GlobeSurfacePointRecord(
            patient_id=self.patient_id,
            study_uid=volume.selection.study_uid,
            series_uid=volume.selection.series_uid,
            modality=modality,
            side=side,
            voxel_zyx=[float(v) for v in native_zyx],
            itk_index_xyz=[float(v) for v in native_index_xyz],
            physical_lps_mm=[float(v) for v in physical_lps],
            view_used=f"dual_view_{panel.view().lower()}_{panel.source().lower()}",
            slice_index_used=float(panel.slice_index()),
            image_spacing_xyz=[float(v) for v in volume.image.GetSpacing()],
            image_origin_lps=[float(v) for v in volume.image.GetOrigin()],
            image_direction_3x3=image_direction_3x3(volume.image),
            source="manual_globe_surface",
            annotator_id=self.annotator_id,
        )

    def move_globe_surface_point(self, panel: DualImagePanel, point: DisplayPoint, row: float, col: float, final: bool) -> None:
        if point.record_id is None or point.side is None:
            return
        record = self.globe_surface_record_from_display(panel, row, col, point.side)
        if record is None:
            return
        self.autosave_disabled_for_patient_id = None
        event_type = "globe_surface_drag_release" if final else "globe_surface_drag"
        self.store.update_globe_surface_point(point.record_id, record, event_type=event_type)
        self.schedule_live_globe_registration_update(final=final)
        if final:
            self.refresh_all_panels()
            self.refresh_3d_panel()
            self.set_status(f"Moved {record.modality} {record.side} globe surface point #{point.record_id}\n{self.status_text()}")

    def move_globe_center_point(self, panel: DualImagePanel, point: DisplayPoint, row: float, col: float, final: bool) -> None:
        if point.side is None:
            return
        if not final:
            return
        record = self.globe_center_override_record_from_display(panel, row, col, point.modality, point.side)
        if record is None:
            return
        self.autosave_disabled_for_patient_id = None
        self.store.upsert_globe_center_override(record)
        self.schedule_live_globe_registration_update(final=True)
        self.refresh_all_panels()
        self.refresh_3d_panel()
        lps_text = ", ".join(f"{value:.2f}" for value in record.physical_lps_mm)
        self.set_status(
            f"Moved forced {self.center_override_label(record.modality, record.side)} to LPS=({lps_text}).\n"
            f"{self.status_text()}"
        )

    def delete_globe_center_override(self, point: DisplayPoint) -> None:
        if point.side is None:
            return
        series_uid = self.series_uid_for_modality(point.modality)
        if not series_uid:
            return
        if point.record_id is None:
            self.set_status(f"{self.center_override_label(point.modality, point.side)} is fitted, not forced; nothing to clear.")
            return
        deleted = self.store.delete_globe_center_override(self.patient_id, point.modality, series_uid, point.side)
        self.autosave_disabled_for_patient_id = None
        self.schedule_live_globe_registration_update(final=True)
        self.refresh_all_panels()
        self.refresh_3d_panel()
        action = "Cleared forced" if deleted else "No forced center found for"
        self.set_status(f"{action} {self.center_override_label(point.modality, point.side)}.\n{self.status_text()}")

    def move_display_point(self, panel: DualImagePanel, point: DisplayPoint, row: float, col: float, final: bool) -> None:
        if not point.editable:
            return
        if point.kind == "globe":
            self.move_globe_surface_point(panel, point, row, col, final)
            return
        if point.kind == "globe_center":
            self.move_globe_center_point(panel, point, row, col, final)
            return
        self.set_current_label(point.label, echo=False)
        event_type = "drag_release" if final else "drag"
        self.save_panel_point(
            panel=panel,
            row=row,
            col=col,
            modality=point.modality,
            label=point.label,
            event_type=event_type,
            refresh=True,
            echo=final,
        )

    def delete_display_point(self, panel: DualImagePanel, point: DisplayPoint) -> None:
        if not point.editable:
            return
        if point.kind == "globe":
            if point.record_id is None:
                return
            self.autosave_disabled_for_patient_id = None
            self.store.delete_globe_surface_point(point.record_id)
            self.schedule_live_globe_registration_update(final=True)
            self.refresh_all_panels()
            self.refresh_3d_panel()
            self.set_status(f"Deleted {point.modality} {point.side} globe surface point #{point.record_id}\n{self.status_text()}")
            return
        if point.kind == "globe_center":
            self.delete_globe_center_override(point)
            return
        series_uid = self.series_uid_for_modality(point.modality)
        if series_uid is None:
            self.set_status(f"Could not delete {point.modality} {point.label}: no active series.")
            return
        self.store.delete_landmark(self.patient_id, point.modality, point.label, series_uid=series_uid)
        self.set_current_label(point.label, echo=False)
        self.refresh_all_panels()
        self.schedule_live_overlay_update(final=True)
        self.set_status(f"Deleted {point.modality} {point.label}\n{self.status_text()}")

    def save_panel_point(
        self,
        panel: DualImagePanel,
        row: float,
        col: float,
        modality: str,
        label: str,
        event_type: str,
        refresh: bool,
        echo: bool,
    ) -> None:
        volume = self.volume_for_modality(modality)
        panel_image = self.panel_image_for_modality(panel, modality)
        if volume is None or panel_image is None:
            self.set_status(f"{modality} point cannot be edited from {panel.source()} view.", echo=echo)
            return
        index_xyz = self.display_to_index_xyz(panel, panel.slice_index(), row, col)
        try:
            physical_lps = tuple(float(v) for v in panel_image.TransformContinuousIndexToPhysicalPoint(index_xyz))
            native_index_xyz = tuple(float(v) for v in volume.image.TransformPhysicalPointToContinuousIndex(physical_lps))
        except Exception as exc:
            self.set_status(f"Could not map click to DICOM physical coordinate: {exc}", echo=echo)
            return
        native_index_xyz = self.clamp_tiny_index_drift(native_index_xyz, volume.image.GetSize())
        physical_lps = tuple(float(v) for v in volume.image.TransformContinuousIndexToPhysicalPoint(native_index_xyz))
        native_zyx = itk_xyz_to_napari_zyx(native_index_xyz)
        if not point_inside_array_shape_zyx(native_zyx, volume.array_zyx.shape):
            self.set_status(f"Rejected {modality} point outside native volume: {native_zyx}", echo=echo)
            return
        existing = self.current_record_for_label(modality, label) if event_type.startswith("drag") else None
        visibility = str(existing.get("visibility", self.current_visibility)) if existing else self.current_visibility
        quality = int(existing.get("quality", self.current_quality)) if existing else self.current_quality
        use_for_transform = bool(existing.get("use_for_transform", visibility in ("visible", "uncertain"))) if existing else visibility in ("visible", "uncertain")
        source = str(existing.get("source", "manual")) if existing else "manual"
        record = LandmarkRecord(
            patient_id=self.patient_id,
            study_uid=volume.selection.study_uid,
            series_uid=volume.selection.series_uid,
            modality=modality,
            landmark_label=label,
            voxel_zyx=[float(v) for v in native_zyx],
            itk_index_xyz=[float(v) for v in native_index_xyz],
            physical_lps_mm=[float(v) for v in physical_lps],
            view_used=f"dual_view_{panel.view().lower()}_{panel.source().lower()}",
            slice_index_used=float(panel.slice_index()),
            image_spacing_xyz=[float(v) for v in volume.image.GetSpacing()],
            image_origin_lps=[float(v) for v in volume.image.GetOrigin()],
            image_direction_3x3=image_direction_3x3(volume.image),
            source=source,
            visibility=visibility,
            use_for_transform=use_for_transform,
            quality=quality,
            annotator_id=self.annotator_id,
        )
        self.store.upsert_landmark(record, event_type=event_type)
        if refresh:
            self.refresh_all_panels()
        self.schedule_live_overlay_update(final=event_type != "drag")
        action = "Moved" if event_type.startswith("drag") else "Saved"
        lps_text = ", ".join(f"{value:.2f}" for value in physical_lps)
        self.set_status(f"{action} {modality} {label} | LPS=({lps_text})\n{self.status_text()}", echo=echo)

    def volume_for_modality(self, modality: str) -> Any | None:
        if modality == "CT":
            return self.ct_volume
        if modality == "MRI":
            return self.mri_volume
        return None

    def current_record_for_label(self, modality: str, label: str) -> dict[str, Any] | None:
        for record in self.fetch_current_records(modality):
            if str(record["landmark_label"]) == str(label):
                return record
        return None

    def panel_image_for_modality(self, panel: DualImagePanel, modality: str) -> Any | None:
        if modality == "CT" and panel.source() == SOURCE_CT:
            return self.ct_lps
        if modality == "MRI" and panel.source() in (SOURCE_MRI, SOURCE_OVERLAY):
            return self.mri_lps
        return None

    @staticmethod
    def clamp_tiny_index_drift(index_xyz: tuple[float, float, float], size_xyz: tuple[int, int, int]) -> tuple[float, float, float]:
        clamped = []
        tolerance = 1.0e-4
        for value, size in zip(index_xyz, size_xyz):
            maximum = float(int(size) - 1)
            value = float(value)
            if -tolerance <= value < 0.0:
                value = 0.0
            elif maximum < value <= maximum + tolerance:
                value = maximum
            clamped.append(value)
        return tuple(clamped)

    def compute_overlay(self) -> None:
        self.update_overlay_preview(write_outputs=True, switch_to_overlay=True, echo=True, refresh=True)

    def schedule_live_overlay_update(self, final: bool) -> None:
        if not self.overlay_preview_requested():
            return
        self.live_overlay_dirty = True
        if final or self.live_overlay_timer is None:
            self.live_overlay_dirty = False
            self.update_overlay_preview(write_outputs=False, switch_to_overlay=False, echo=False, refresh=True)
            return
        if not self.live_overlay_timer.isActive():
            self.live_overlay_timer.start(self.live_overlay_interval_ms)

    def _on_live_overlay_timer(self) -> None:
        if not self.live_overlay_dirty:
            return
        self.live_overlay_dirty = False
        self.update_overlay_preview(write_outputs=False, switch_to_overlay=False, echo=False, refresh=True)

    def overlay_preview_requested(self) -> bool:
        if self.overlay_result is not None:
            return True
        return any(panel is not None and panel.source() == SOURCE_OVERLAY for panel in [self.left_panel, self.right_panel])

    def schedule_live_globe_registration_update(self, final: bool) -> None:
        if not self.globe_preview_requested():
            return
        self.live_globe_dirty = True
        if final or self.live_globe_timer is None:
            self.live_globe_dirty = False
            self.update_globe_registration_preview(write_outputs=False, switch_to_preview=False, echo=False)
            return
        if not self.live_globe_timer.isActive():
            self.live_globe_timer.start(self.live_overlay_interval_ms)

    def _on_live_globe_timer(self) -> None:
        if not self.live_globe_dirty:
            return
        self.live_globe_dirty = False
        self.update_globe_registration_preview(write_outputs=False, switch_to_preview=False, echo=False)

    def globe_preview_requested(self) -> bool:
        if self.globe_registration_result is not None:
            return True
        return any(panel is not None and panel.source() == SOURCE_MRI_ON_CT for panel in [self.left_panel, self.right_panel])

    def update_overlay_preview(self, write_outputs: bool, switch_to_overlay: bool, echo: bool, refresh: bool) -> bool:
        if self.ct_volume is None or self.mri_volume is None:
            self.set_status("Volumes are not ready.", echo=echo)
            return False
        records = self.current_landmark_records()
        try:
            result = estimate_rigid_initializer(records, fixed_modality="MRI", moving_modality="CT")
        except Exception as exc:
            self.set_status(f"Initialization failed: {exc}\n{self.status_text()}", echo=echo)
            return False
        out_dir = self.initialization_output_dir()
        if write_outputs:
            save_initialization_result(result, out_dir)
            resampled_ct = write_resampled_moving_to_fixed(
                moving_image=self.ct_volume.image,
                fixed_image=self.mri_volume.image,
                transform_fixed_to_moving=result.transform_fixed_to_moving,
                output_path=out_dir / "ct_resampled_to_mri_init.nii.gz",
                default_value=-1024.0,
            )
        else:
            resampled_ct = resample_moving_to_fixed(
                moving_image=self.ct_volume.image,
                fixed_image=self.mri_volume.image,
                transform_fixed_to_moving=result.transform_fixed_to_moving,
                default_value=-1024.0,
            )
        resampled_lps = self.orient_to_lps(resampled_ct)
        self.overlay_result = result
        self.overlay_ct_soft_lps = ct_window(self.sitk_array(resampled_lps), -100, 200)
        self.overlay_ct_bone_lps = ct_window(self.sitk_array(resampled_lps), -500, 1500)
        if switch_to_overlay:
            overlay_targets = [
                panel
                for panel in [self.left_panel, self.right_panel]
                if panel is not None and panel.source() == SOURCE_MRI
            ]
            if not overlay_targets and self.right_panel is not None:
                overlay_targets = [self.right_panel]
            for panel in overlay_targets:
                panel.set_source(SOURCE_OVERLAY)
        if refresh:
            self.refresh_all_panels()
        summary = result_summary(result)
        saved_text = f"\nsaved={out_dir}" if write_outputs else "\nlive preview only; press Shift to save transform/QC files"
        self.set_status(
            f"Overlay ready for selected MRI: {self.mri_series_description}\n"
            f"status={summary['status']} median={summary['median_residual_mm']:.2f} mm "
            f"max={summary['max_residual_mm']:.2f} mm"
            f"{saved_text}",
            echo=echo,
        )
        return True

    def update_globe_registration_preview(self, write_outputs: bool, switch_to_preview: bool, echo: bool) -> bool:
        if self.ct_volume is None or self.mri_volume is None:
            self.set_status("Volumes are not ready.", echo=echo)
            return False
        self.save_manual_parameters_for_current_series()
        fits = self.current_globe_sphere_fits()
        fixed_fits = {side: fits[("CT", side)] for side in ("L", "R") if ("CT", side) in fits}
        moving_fits = {side: fits[("MRI", side)] for side in ("L", "R") if ("MRI", side) in fits}
        try:
            result = estimate_globe_manual_initializer(
                fixed_fits=fixed_fits,
                moving_fits=moving_fits,
                pitch_deg=self.pitch_degrees(),
                scale_xyz=self.manual_scale_xyz(),
            )
        except Exception as exc:
            self.set_status(
                f"Globe MRI-on-CT initialization failed: {exc}\n"
                f"{self.globe_fit_status_text()}",
                echo=echo,
            )
            return False
        resampled = resample_mri_to_ct(
            moving_mri=self.mri_volume.image,
            fixed_ct=self.ct_volume.image,
            transform_fixed_to_moving=result.transform_fixed_to_moving,
            default_value=0.0,
        )
        resampled_lps = self.orient_to_lps(resampled)
        self.globe_registration_result = result
        self.globe_mri_on_ct_lps = resampled_lps
        self.globe_mri_on_ct_display_lps = normalize_percentile(self.sitk_array(resampled_lps))
        out_dir = self.globe_registration_output_dir()
        if write_outputs:
            import SimpleITK as sitk

            save_globe_manual_registration(result, out_dir)
            sitk.WriteImage(resampled, str(out_dir / "mri_resampled_to_ct_globe_manual.nii.gz"))
        if switch_to_preview:
            target = self.left_panel or self.right_panel
            if target is not None:
                target.set_source(SOURCE_MRI_ON_CT)
        self.refresh_all_panels()
        self.refresh_3d_panel()
        saved = f"\nsaved={out_dir}" if write_outputs else "\nlive preview only; use Save globe transform to write files"
        self.set_status(
            f"Globe MRI-on-CT ready | pitch={result.pitch_deg:.2f}, "
            f"scale=({result.scale_xyz[0]:.3f}, {result.scale_xyz[1]:.3f}, {result.scale_xyz[2]:.3f})\n"
            f"eye distance CT={result.eye_distance_fixed_mm:.2f} mm, MRI={result.eye_distance_moving_mm:.2f} mm"
            f"{saved}\n{self.globe_fit_status_text()}",
            echo=echo,
        )
        return True

    def autosave_current_patient_before_switch(self) -> None:
        if self.ct_volume is None or self.mri_volume is None:
            return
        if self.autosave_disabled_for_patient_id == str(self.patient_id):
            self.set_status(f"Autosave skipped for deleted patient {self.patient_id}.", echo=True)
            return
        self.save_manual_parameters_for_current_series()
        fits = self.current_globe_sphere_fits()
        required = {("CT", "L"), ("CT", "R"), ("MRI", "L"), ("MRI", "R")}
        if not required.issubset(fits):
            self.set_status(
                f"Autosaved parameters for {self.patient_id}; transform not written because fitted CT/MRI L/R spheres are incomplete.\n"
                f"{self.status_text()}",
                echo=True,
            )
            return
        try:
            self.update_globe_registration_preview(write_outputs=True, switch_to_preview=False, echo=True)
        except Exception as exc:
            self.set_status(
                f"Autosaved parameters for {self.patient_id}, but final transform autosave failed: {exc}\n"
                f"{self.status_text()}",
                echo=True,
            )

    def pitch_degrees(self) -> float:
        return float(self.pitch_spin.value()) if self.pitch_spin is not None else 0.0

    def manual_scale_xyz(self) -> tuple[float, float, float]:
        return (
            float(self.scale_x_spin.value()) if self.scale_x_spin is not None else 1.0,
            float(self.scale_y_spin.value()) if self.scale_y_spin is not None else 1.0,
            float(self.scale_z_spin.value()) if self.scale_z_spin is not None else 1.0,
        )

    def on_manual_parameter_changed(self) -> None:
        if self.loading_manual_parameters:
            return
        self.autosave_disabled_for_patient_id = None
        self.save_manual_parameters_for_current_series()
        self.update_globe_registration_preview(write_outputs=False, switch_to_preview=False, echo=False)

    def save_manual_parameters_for_current_series(self) -> None:
        if self.ct_volume is None or self.mri_volume is None:
            return
        ct_uid = self.series_uid_for_modality("CT")
        mri_uid = self.series_uid_for_modality("MRI")
        if not ct_uid or not mri_uid:
            return
        self.store.upsert_globe_manual_parameters(
            patient_id=self.patient_id,
            ct_series_uid=ct_uid,
            mri_series_uid=mri_uid,
            mri_series_description=self.mri_series_description,
            pitch_deg=self.pitch_degrees(),
            scale_xyz=self.manual_scale_xyz(),
            annotator_id=self.annotator_id,
        )

    def load_manual_parameters_for_current_series(self) -> None:
        if any(spin is None for spin in [self.pitch_spin, self.scale_x_spin, self.scale_y_spin, self.scale_z_spin]):
            return
        if self.ct_volume is None or self.mri_volume is None:
            return
        ct_uid = self.series_uid_for_modality("CT")
        mri_uid = self.series_uid_for_modality("MRI")
        if not ct_uid or not mri_uid:
            return
        params = self.store.fetch_globe_manual_parameters(self.patient_id, ct_uid, mri_uid)
        if params is None:
            params = self.read_saved_globe_manual_parameters()
            source = "file"
        else:
            source = "db"
        if params is None:
            pitch = 0.0
            scale = (1.0, 1.0, 1.0)
            source = "default"
        else:
            pitch = float(params.get("pitch_deg", 0.0))
            scale_values = params.get("scale_xyz", [1.0, 1.0, 1.0])
            scale = tuple(float(v) for v in list(scale_values)[:3])
            if len(scale) != 3:
                scale = (1.0, 1.0, 1.0)
        self.set_manual_parameter_spinboxes(pitch, scale)
        if source == "file":
            self.save_manual_parameters_for_current_series()

    def read_saved_globe_manual_parameters(self) -> dict[str, Any] | None:
        path = self.globe_registration_output_dir() / "globe_manual_registration_qc.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if "pitch_deg" not in payload or "scale_xyz" not in payload:
            return None
        return payload

    def set_manual_parameter_spinboxes(self, pitch_deg: float, scale_xyz: tuple[float, float, float]) -> None:
        spins = [
            (self.pitch_spin, float(pitch_deg)),
            (self.scale_x_spin, float(scale_xyz[0])),
            (self.scale_y_spin, float(scale_xyz[1])),
            (self.scale_z_spin, float(scale_xyz[2])),
        ]
        self.loading_manual_parameters = True
        try:
            for spin, value in spins:
                if spin is None:
                    continue
                previous = spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(previous)
        finally:
            self.loading_manual_parameters = False

    def globe_registration_output_dir(self) -> Path:
        return self.globe_registration_output_dir_for_patient(self.patient_id, self.mri_series_description)

    def globe_registration_output_dir_for_patient(self, patient_id: str, series_description: str) -> Path:
        return self.patient_output_dir(patient_id) / "globe_manual_registration" / self.safe_path_component(series_description)

    def patient_output_dir(self, patient_id: str | None = None) -> Path:
        return self.db_path.parent / str(patient_id if patient_id is not None else self.patient_id)

    def globe_fit_status_text(self) -> str:
        fits = self.current_globe_sphere_fits()
        counts = {
            (modality, side): len(self.store.fetch_globe_surface_points(
                self.patient_id,
                modality,
                series_uid=self.series_uid_for_modality(modality),
                side=side,
            ))
            for modality in ("CT", "MRI")
            for side in ("L", "R")
            if self.series_uid_for_modality(modality) is not None
        }
        parts = []
        for modality in ("CT", "MRI"):
            for side in ("L", "R"):
                fit = fits.get((modality, side))
                count = counts.get((modality, side), 0)
                if fit:
                    forced = ", forced center" if getattr(fit, "status", "") == "manual_override" else ""
                    parts.append(f"{modality}-{side}: n={fit.n_points}, r={fit.radius_mm:.1f}, rms={fit.rms_residual_mm:.2f}{forced}")
                else:
                    parts.append(f"{modality}-{side}: n={count}, no sphere")
        return " | ".join(parts)

    def display_points_for_panel(self, panel: DualImagePanel) -> list[DisplayPoint]:
        target_image = self.image_for_panel(panel)
        if target_image is None:
            return []
        return self.display_globe_points_for_panel(panel, target_image)

    def display_landmark_points_for_panel(self, panel: DualImagePanel, target_image: Any) -> list[DisplayPoint]:
        source = panel.source()
        records: list[tuple[dict[str, Any], str, tuple[float, float, float] | None, bool]] = []
        if source == SOURCE_CT:
            for record in self.fetch_current_records("CT"):
                records.append((record, "#ffd21f", None, True))
        elif source == SOURCE_MRI:
            for record in self.fetch_current_records("MRI"):
                records.append((record, "#22d3ee", None, True))
        elif source == SOURCE_OVERLAY:
            for record in self.fetch_current_records("MRI"):
                records.append((record, "#22d3ee", None, True))
            if self.overlay_result is not None:
                for record in self.fetch_current_records("CT"):
                    ct_point = tuple(float(v) for v in record["physical_lps_mm"])
                    projected = self.overlay_result.transform_moving_to_fixed.TransformPoint(ct_point)
                    records.append((record, "#ff4040", tuple(float(v) for v in projected), False))
        elif source == SOURCE_MRI_ON_CT:
            for record in self.fetch_current_records("CT"):
                records.append((record, "#ffd21f", None, True))
            if self.globe_registration_result is not None:
                for record in self.fetch_current_records("MRI"):
                    mri_point = tuple(float(v) for v in record["physical_lps_mm"])
                    projected = self.globe_registration_result.transform_moving_to_fixed.TransformPoint(mri_point)
                    records.append((record, "#22d3ee", tuple(float(v) for v in projected), False))
        points: list[DisplayPoint] = []
        for record, color, projected_lps, editable in records:
            point_lps = projected_lps or tuple(float(v) for v in record["physical_lps_mm"])
            try:
                point_zyx = napari_zyx_from_physical_lps(target_image, point_lps)
            except Exception:
                continue
            display = self.zyx_to_display(panel.view(), point_zyx)
            if display is None:
                continue
            slice_pos, row, col = display
            if abs(slice_pos - panel.slice_index()) <= 0.75:
                display_row = self.original_row_to_display_row(panel, row, self.raw_slice_height(panel))
                display_col = self.original_col_to_display_col(panel, col, self.raw_slice_width(panel))
                points.append(
                    DisplayPoint(
                        row=display_row,
                        col=display_col,
                        color=color,
                        label=str(record["landmark_label"]),
                        modality=str(record["modality"]).upper(),
                        editable=editable,
                    )
                )
        return points

    def display_globe_points_for_panel(self, panel: DualImagePanel, target_image: Any) -> list[DisplayPoint]:
        source = panel.source()
        display_items: list[tuple[dict[str, Any], tuple[float, float, float], str, bool, str]] = []
        if source == SOURCE_CT:
            for record in self.fetch_current_globe_points("CT"):
                display_items.append((record, tuple(float(v) for v in record["physical_lps_mm"]), "#65ff6b", True, f"G{record['side']}"))
        elif source in (SOURCE_MRI, SOURCE_OVERLAY):
            for record in self.fetch_current_globe_points("MRI"):
                display_items.append((record, tuple(float(v) for v in record["physical_lps_mm"]), "#ff6bd6", True, f"G{record['side']}"))
        elif source == SOURCE_MRI_ON_CT:
            for record in self.fetch_current_globe_points("CT"):
                display_items.append((record, tuple(float(v) for v in record["physical_lps_mm"]), "#65ff6b", False, f"CT G{record['side']}"))
            if self.globe_registration_result is not None:
                for record in self.fetch_current_globe_points("MRI"):
                    moved = self.globe_registration_result.transform_moving_to_fixed.TransformPoint(
                        tuple(float(v) for v in record["physical_lps_mm"])
                    )
                    display_items.append((record, tuple(float(v) for v in moved), "#ff6bd6", False, f"MRI G{record['side']}"))

        points: list[DisplayPoint] = []
        for record, point_lps, color, editable, label in display_items:
            point = self.display_point_from_lps(
                panel=panel,
                target_image=target_image,
                point_lps=point_lps,
                color=color,
                label=label,
                modality=str(record["modality"]).upper(),
                editable=editable,
                kind="globe",
                record_id=int(record["id"]) if "id" in record else None,
                side=str(record.get("side", "")),
            )
            if point is not None:
                points.append(point)

        fits = self.current_globe_sphere_fits()
        overrides = {
            (str(record["modality"]).upper(), str(record["side"]).upper()): record
            for record in self.fetch_current_globe_center_overrides()
        }
        center_items: list[tuple[tuple[float, float, float], str, str, str, str, bool, int | None]] = []
        if source == SOURCE_CT:
            for side in ("L", "R"):
                fit = fits.get(("CT", side))
                if fit:
                    override = overrides.get(("CT", side))
                    label = self.center_override_label("CT", side) + ("*" if override else "")
                    center_items.append((tuple(fit.center_lps), "#ff9f1c" if override else "#eaff00", label, "CT", side, True, int(override["id"]) if override else None))
        elif source in (SOURCE_MRI, SOURCE_OVERLAY):
            for side in ("L", "R"):
                fit = fits.get(("MRI", side))
                if fit:
                    override = overrides.get(("MRI", side))
                    label = self.center_override_label("MRI", side) + ("*" if override else "")
                    editable = source == SOURCE_MRI
                    center_items.append((tuple(fit.center_lps), "#ff9f1c" if override else "#00f0ff", label, "MRI", side, editable, int(override["id"]) if override else None))
        elif source == SOURCE_MRI_ON_CT:
            for side in ("L", "R"):
                fit = fits.get(("CT", side))
                if fit:
                    override = overrides.get(("CT", side))
                    label = self.center_override_label("CT", side) + ("*" if override else "")
                    center_items.append((tuple(fit.center_lps), "#ff9f1c" if override else "#eaff00", label, "CT", side, False, int(override["id"]) if override else None))
                moving_fit = fits.get(("MRI", side))
                if moving_fit and self.globe_registration_result is not None:
                    moved = self.globe_registration_result.transform_moving_to_fixed.TransformPoint(tuple(moving_fit.center_lps))
                    override = overrides.get(("MRI", side))
                    label = self.center_override_label("MRI", side) + ("*" if override else "")
                    center_items.append((tuple(float(v) for v in moved), "#ff9f1c" if override else "#00f0ff", label, "MRI", side, False, int(override["id"]) if override else None))
        for point_lps, color, label, modality, side, editable, record_id in center_items:
            point = self.display_point_from_lps(
                panel=panel,
                target_image=target_image,
                point_lps=point_lps,
                color=color,
                label=label,
                modality=modality,
                editable=editable,
                kind="globe_center",
                record_id=record_id,
                side=side,
            )
            if point is not None:
                points.append(point)
        return points

    def display_point_from_lps(
        self,
        panel: DualImagePanel,
        target_image: Any,
        point_lps: tuple[float, float, float],
        color: str,
        label: str,
        modality: str,
        editable: bool,
        kind: str,
        record_id: int | None = None,
        side: str | None = None,
    ) -> DisplayPoint | None:
        try:
            point_zyx = napari_zyx_from_physical_lps(target_image, point_lps)
        except Exception:
            return None
        display = self.zyx_to_display(panel.view(), point_zyx)
        if display is None:
            return None
        slice_pos, row, col = display
        if abs(slice_pos - panel.slice_index()) > 0.75:
            return None
        return DisplayPoint(
            row=self.original_row_to_display_row(panel, row, self.raw_slice_height(panel)),
            col=self.original_col_to_display_col(panel, col, self.raw_slice_width(panel)),
            color=color,
            label=label,
            modality=modality,
            editable=editable,
            kind=kind,
            record_id=record_id,
            side=side,
        )

    def current_landmark_records(self) -> list[dict[str, Any]]:
        records = []
        records.extend(self.fetch_current_records("CT"))
        records.extend(self.fetch_current_records("MRI"))
        return records

    def fetch_current_records(self, modality: str) -> list[dict[str, Any]]:
        uid = self.series_uid_for_modality(modality)
        if uid is None:
            return []
        return self.store.fetch_landmarks(self.patient_id, modality, series_uid=uid)

    def fetch_current_globe_points(self, modality: str) -> list[dict[str, Any]]:
        uid = self.series_uid_for_modality(modality)
        if uid is None:
            return []
        return self.store.fetch_globe_surface_points(self.patient_id, modality, series_uid=uid)

    def fetch_current_globe_center_overrides(self, modality: str | None = None) -> list[dict[str, Any]]:
        if modality is not None:
            uid = self.series_uid_for_modality(modality)
            if uid is None:
                return []
            return self.store.fetch_globe_center_overrides(self.patient_id, modality, series_uid=uid)
        records: list[dict[str, Any]] = []
        for item in ("CT", "MRI"):
            records.extend(self.fetch_current_globe_center_overrides(item))
        return records

    def current_globe_sphere_fits(self) -> dict[tuple[str, str], Any]:
        points: list[dict[str, Any]] = []
        points.extend(self.fetch_current_globe_points("CT"))
        points.extend(self.fetch_current_globe_points("MRI"))
        fits = fit_globe_spheres(points)
        return self.apply_globe_center_overrides(fits)

    def apply_globe_center_overrides(self, fits: dict[tuple[str, str], SphereFit]) -> dict[tuple[str, str], SphereFit]:
        for override in self.fetch_current_globe_center_overrides():
            key = (str(override["modality"]).upper(), str(override["side"]).upper())
            previous = fits.get(key)
            if previous is not None:
                radius = float(previous.radius_mm)
                n_points = int(previous.n_points)
                rms = float(previous.rms_residual_mm)
                maximum = float(previous.max_residual_mm)
            else:
                radius = self.default_globe_radius_mm(fits, key[0])
                n_points = 0
                rms = 0.0
                maximum = 0.0
            fits[key] = SphereFit(
                center_lps=[float(v) for v in override["physical_lps_mm"]],
                radius_mm=radius,
                n_points=n_points,
                rms_residual_mm=rms,
                max_residual_mm=maximum,
                status="manual_override",
            )
        return fits

    @staticmethod
    def default_globe_radius_mm(fits: dict[tuple[str, str], SphereFit], modality: str) -> float:
        radii = [float(fit.radius_mm) for (fit_modality, _side), fit in fits.items() if fit_modality == modality and fit.radius_mm > 0.0]
        if not radii:
            radii = [float(fit.radius_mm) for fit in fits.values() if fit.radius_mm > 0.0]
        return float(np.median(radii)) if radii else 12.0

    def series_uid_for_modality(self, modality: str) -> str | None:
        if modality == "CT" and self.ct_volume is not None:
            return self.ct_volume.selection.series_uid
        if modality == "MRI" and self.mri_volume is not None:
            return self.mri_volume.selection.series_uid
        return None

    def refresh_3d_panel(self) -> None:
        if self.three_d_panel is not None:
            self.three_d_panel.update_scene()

    def set_3d_mode(self, mode: str) -> None:
        if self.three_d_panel is not None:
            self.three_d_panel.set_mode(mode)
            self.set_status(f"3D mode: {mode}\n{self.status_text()}")

    def set_active_zoom_panel(self, panel: DualImagePanel) -> None:
        self.active_zoom_panel = panel

    def zoom_active_panel(self, factor: float) -> None:
        panel = self.active_zoom_panel or self.right_panel or self.left_panel
        if panel is None:
            return
        panel.zoom_by(factor)

    def reset_active_panel_zoom(self) -> None:
        panel = self.active_zoom_panel or self.right_panel or self.left_panel
        if panel is None:
            return
        panel.reset_zoom()

    def set_globe_side(self, side: str, echo: bool = True) -> None:
        value = str(side).upper()
        if value not in ("L", "R"):
            self.set_status(f"Unknown globe side: {side}", echo=echo)
            return
        if self.globe_side_combo is not None and self.globe_side_combo.currentText() != value:
            self.globe_side_combo.setCurrentText(value)
        if echo:
            self.set_status(f"Manual globe surface side: {value} (used for sagittal/fallback)\n{self.status_text()}", echo=True)

    def set_current_label(self, label: str, echo: bool = True) -> None:
        self.current_label = label
        if self.label_combo is not None:
            self.label_combo.setCurrentText(label)
        self.set_status(f"Active landmark: {label}", echo=echo)

    def status_text(self) -> str:
        counts = {
            (modality, side): len(self.store.fetch_globe_surface_points(
                self.patient_id,
                modality,
                series_uid=self.series_uid_for_modality(modality),
                side=side,
            ))
            for modality in ("CT", "MRI")
            for side in ("L", "R")
            if self.series_uid_for_modality(modality) is not None
        }
        return (
            f"patient={self.patient_id} queue={self.queue_position_text()} | "
            f"{self.registration_progress_text()} | "
            f"MRI={self.mri_series_description} | "
            f"eye side=auto axial/coronal, manual {self.globe_side()} fallback | "
            f"pitch={self.pitch_degrees():.2f}, "
            f"scale=({self.manual_scale_xyz()[0]:.3f}, {self.manual_scale_xyz()[1]:.3f}, {self.manual_scale_xyz()[2]:.3f}) | "
            f"globe surface CT L/R={counts.get(('CT', 'L'), 0)}/{counts.get(('CT', 'R'), 0)}, "
            f"MRI L/R={counts.get(('MRI', 'L'), 0)}/{counts.get(('MRI', 'R'), 0)}\n"
            f"{self.globe_fit_status_text()}\n"
            f"{self.mri_availability_text()}"
        )

    def mri_availability_text(self) -> str:
        rows = self.mri_candidate_rows_for_patient(self.patient_id)
        parts = []
        for description in MRI_CANDIDATE_SERIES:
            short_name = MRI_CANDIDATE_SHORT_NAMES.get(description, description)
            row = rows.get(description)
            current = "*" if description == self.mri_series_description else ""
            parts.append(f"{short_name}:{row.get('instance_count') if row else '-'}{current}")
        return " | ".join(parts)

    def set_status(self, text: str, echo: bool = True) -> None:
        self.update_progress_label()
        if self.status_label is not None:
            self.status_label.setText(text)
        if echo:
            print(text)

    def _load_ct_volume(self, patient_id: str, queue_row: dict[str, str] | None) -> Any:
        if queue_row and queue_row.get("ct_series_uid"):
            return load_series_volume(selection_from_queue_row(queue_row, "CT"))
        return load_series_volume(select_series(self.manifest_csv, patient_id, "CT", self.ct_series_description))

    def _load_mri_volume(self, patient_id: str, series_description: str, queue_row: dict[str, str] | None) -> Any:
        row = self._mri_candidate_row(patient_id, series_description)
        if row:
            return load_series_volume(selection_from_manifest_row(row, "MR"))
        if queue_row and queue_row.get("mri_series_description") == series_description and queue_row.get("mri_series_uid"):
            return load_series_volume(selection_from_queue_row(queue_row, "MRI"))
        raise RuntimeError(f"Patient {patient_id} has no MRI series: {series_description}")

    def _mri_candidate_row(self, patient_id: str, series_description: str) -> dict[str, str] | None:
        return self.mri_candidate_rows_for_patient(patient_id).get(series_description)

    def mri_candidate_rows_for_patient(self, patient_id: str) -> dict[str, dict[str, str]]:
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
            if previous is None or self.series_sort_key(row) > self.series_sort_key(previous):
                candidates[description] = row
        self._mri_candidate_cache[patient_id] = candidates
        return candidates

    def initialization_output_dir(self) -> Path:
        return self.db_path.parent / self.patient_id / "initialization" / self.safe_path_component(self.mri_series_description)

    @staticmethod
    def series_sort_key(row: dict[str, str]) -> tuple[int, str, str]:
        try:
            count = int(float(row.get("instance_count") or 0))
        except ValueError:
            count = 0
        return (count, str(row.get("study_folder") or ""), str(row.get("series_uid") or ""))

    def display_slice(self, panel: DualImagePanel, array: np.ndarray, index: int) -> np.ndarray:
        raw = self.extract_slice(array, panel.view(), index)
        resized = self.resize_rows(raw, self.display_slice_height(panel, raw.shape[0]))
        resized = self.resize_columns(resized, self.display_slice_width(panel, raw.shape[1]))
        if self.flip_display_vertical(panel):
            resized = np.flipud(resized)
        return resized

    def display_slice_height(self, panel: DualImagePanel, raw_height: int) -> int:
        if panel.view() == VIEW_AXIAL and panel.source() in (SOURCE_MRI, SOURCE_OVERLAY):
            ct_slice_count = int(self.ct_soft_lps.shape[0]) if self.ct_soft_lps is not None else 0
            return max(int(raw_height), ct_slice_count)
        return int(raw_height)

    def raw_slice_height(self, panel: DualImagePanel) -> int:
        array = self.array_for_panel(panel)
        if array is None:
            return 0
        if panel.view() == VIEW_AXIAL:
            return int(array.shape[1])
        return int(array.shape[0])

    def display_slice_width(self, panel: DualImagePanel, raw_width: int) -> int:
        if panel.view() == VIEW_SAGITTAL and panel.source() in (SOURCE_MRI, SOURCE_OVERLAY):
            image = self.image_for_panel(panel)
            if image is not None:
                spacing_xyz = tuple(float(v) for v in image.GetSpacing())
                row_spacing = spacing_xyz[2]
                col_spacing = spacing_xyz[1]
                if row_spacing > 0.0 and col_spacing > 0.0:
                    return max(int(raw_width), int(round(float(raw_width) * col_spacing / row_spacing)))
            ct_slice_count = int(self.ct_soft_lps.shape[0]) if self.ct_soft_lps is not None else 0
            return max(int(raw_width), ct_slice_count)
        return int(raw_width)

    def raw_slice_width(self, panel: DualImagePanel) -> int:
        array = self.array_for_panel(panel)
        if array is None:
            return 0
        if panel.view() == VIEW_AXIAL:
            return int(array.shape[2])
        if panel.view() == VIEW_CORONAL:
            return int(array.shape[2])
        if panel.view() == VIEW_SAGITTAL:
            return int(array.shape[1])
        return 0

    @staticmethod
    def flip_display_vertical(panel: DualImagePanel) -> bool:
        return panel.view() in (VIEW_CORONAL, VIEW_SAGITTAL)

    @staticmethod
    def resize_rows(image: np.ndarray, target_height: int) -> np.ndarray:
        source_height = int(image.shape[0])
        target_height = int(target_height)
        if target_height <= 0 or target_height == source_height:
            return image
        if source_height <= 1:
            return np.repeat(image[:1, :], target_height, axis=0)
        positions = np.linspace(0.0, float(source_height - 1), target_height, dtype=np.float32)
        low = np.floor(positions).astype(np.int32)
        high = np.minimum(low + 1, source_height - 1)
        weight = (positions - low).astype(np.float32)[:, None]
        return image[low, :] * (1.0 - weight) + image[high, :] * weight

    @staticmethod
    def resize_columns(image: np.ndarray, target_width: int) -> np.ndarray:
        source_width = int(image.shape[1])
        target_width = int(target_width)
        if target_width <= 0 or target_width == source_width:
            return image
        if source_width <= 1:
            return np.repeat(image[:, :1], target_width, axis=1)
        positions = np.linspace(0.0, float(source_width - 1), target_width, dtype=np.float32)
        low = np.floor(positions).astype(np.int32)
        high = np.minimum(low + 1, source_width - 1)
        weight = (positions - low).astype(np.float32)[None, :]
        return image[:, low] * (1.0 - weight) + image[:, high] * weight

    def display_row_to_original_row(self, panel: DualImagePanel, row: float, raw_height: int) -> float:
        display_height = self.display_slice_height(panel, raw_height)
        mapped = float(row)
        if self.flip_display_vertical(panel):
            mapped = float(display_height - 1) - mapped
        if display_height > 1 and raw_height > 1 and display_height != raw_height:
            mapped = mapped * float(raw_height - 1) / float(display_height - 1)
        return mapped

    def original_row_to_display_row(self, panel: DualImagePanel, row: float, raw_height: int) -> float:
        display_height = self.display_slice_height(panel, raw_height)
        mapped = float(row)
        if display_height > 1 and raw_height > 1 and display_height != raw_height:
            mapped = mapped * float(display_height - 1) / float(raw_height - 1)
        if self.flip_display_vertical(panel):
            mapped = float(display_height - 1) - mapped
        return mapped

    def display_col_to_original_col(self, panel: DualImagePanel, col: float, raw_width: int) -> float:
        display_width = self.display_slice_width(panel, raw_width)
        mapped = float(col)
        if display_width > 1 and raw_width > 1 and display_width != raw_width:
            mapped = mapped * float(raw_width - 1) / float(display_width - 1)
        return mapped

    def original_col_to_display_col(self, panel: DualImagePanel, col: float, raw_width: int) -> float:
        display_width = self.display_slice_width(panel, raw_width)
        mapped = float(col)
        if display_width > 1 and raw_width > 1 and display_width != raw_width:
            mapped = mapped * float(display_width - 1) / float(raw_width - 1)
        return mapped

    @staticmethod
    def slice_axis(view: str) -> int:
        if view == VIEW_AXIAL:
            return 0
        if view == VIEW_CORONAL:
            return 1
        if view == VIEW_SAGITTAL:
            return 2
        raise ValueError(f"Unknown view: {view}")

    @staticmethod
    def extract_slice(array: np.ndarray, view: str, index: int) -> np.ndarray:
        if view == VIEW_AXIAL:
            return array[index, :, :]
        if view == VIEW_CORONAL:
            return array[:, index, :]
        if view == VIEW_SAGITTAL:
            return array[:, :, index]
        raise ValueError(f"Unknown view: {view}")

    def display_to_index_xyz(self, panel: DualImagePanel, slice_index: int, row: float, col: float) -> tuple[float, float, float]:
        raw_row = self.display_row_to_original_row(panel, row, self.raw_slice_height(panel))
        raw_col = self.display_col_to_original_col(panel, col, self.raw_slice_width(panel))
        if panel.view() == VIEW_AXIAL:
            return (float(raw_col), float(raw_row), float(slice_index))
        if panel.view() == VIEW_CORONAL:
            return (float(raw_col), float(slice_index), float(raw_row))
        if panel.view() == VIEW_SAGITTAL:
            return (float(slice_index), float(raw_col), float(raw_row))
        raise ValueError(f"Unknown view: {panel.view()}")

    @staticmethod
    def zyx_to_display(view: str, point_zyx: tuple[float, float, float]) -> tuple[float, float, float] | None:
        z, y, x = point_zyx
        if view == VIEW_AXIAL:
            return (z, y, x)
        if view == VIEW_CORONAL:
            return (y, z, x)
        if view == VIEW_SAGITTAL:
            return (x, z, y)
        return None

    @staticmethod
    def gray_rgb(image: np.ndarray) -> np.ndarray:
        gray = np.clip(image.astype(np.float32), 0.0, 1.0)
        rgb = (gray * 255.0).astype(np.uint8)
        return np.stack([rgb, rgb, rgb], axis=-1)

    @staticmethod
    def overlay_rgb(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        base_rgb = DualViewWorkbench.gray_rgb(base).astype(np.float32)
        alpha = np.clip(overlay.astype(np.float32), 0.0, 1.0) * 0.65
        color = np.zeros_like(base_rgb)
        color[..., 0] = 255.0
        color[..., 1] = 80.0
        out = base_rgb * (1.0 - alpha[..., None]) + color * alpha[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def overlay_rgb_color(base: np.ndarray, overlay: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
        base_rgb = DualViewWorkbench.gray_rgb(base).astype(np.float32)
        alpha = np.clip(overlay.astype(np.float32), 0.0, 1.0) * 0.6
        color_arr = np.zeros_like(base_rgb)
        color_arr[..., 0] = float(color[0])
        color_arr[..., 1] = float(color[1])
        color_arr[..., 2] = float(color[2])
        out = base_rgb * (1.0 - alpha[..., None]) + color_arr * alpha[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def orient_to_lps(image: Any) -> Any:
        import SimpleITK as sitk

        return sitk.DICOMOrient(image, "LPS")

    @staticmethod
    def image_center_lps_x(image: Any) -> float:
        size = image.GetSize()
        center_index = tuple((float(size[axis]) - 1.0) / 2.0 for axis in range(3))
        return float(image.TransformContinuousIndexToPhysicalPoint(center_index)[0])

    @staticmethod
    def sitk_array(image: Any) -> np.ndarray:
        import SimpleITK as sitk

        return sitk.GetArrayFromImage(image)

    @staticmethod
    def safe_path_component(value: str) -> str:
        safe = "".join(ch if ch.isalnum() else "_" for ch in value)
        while "__" in safe:
            safe = safe.replace("__", "_")
        return safe.strip("_") or "series"

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

    def queue_position_text(self) -> str:
        if not self.queue_rows:
            return "none"
        if self.queue_index is None:
            return f"not in queue / {len(self.queue_rows)}"
        return f"{self.queue_index + 1}/{len(self.queue_rows)}"

    def registration_progress_patient_ids(self) -> list[str]:
        patients: list[str] = []
        seen: set[str] = set()
        if self.queue_rows:
            for row in self.queue_rows:
                patient_id = str(row.get("patient_id", "")).strip()
                if not patient_id or patient_id in seen:
                    continue
                row_series = str(row.get("mri_series_description", "")).strip()
                if row_series and row_series != self.mri_series_description:
                    continue
                patients.append(patient_id)
                seen.add(patient_id)
            return patients
        for patient_id in sorted(
            {
                str(row.get("patient_id", "")).strip()
                for row in self.manifest_rows
                if str(row.get("modality", "")).upper() == "MR"
                and str(row.get("series_description", "")) == self.mri_series_description
                and str(row.get("patient_id", "")).strip()
            }
        ):
            if patient_id not in seen:
                patients.append(patient_id)
                seen.add(patient_id)
        return patients

    def registration_progress_counts(self) -> tuple[int, int]:
        patients = self.registration_progress_patient_ids()
        completed = 0
        for patient_id in patients:
            qc_path = self.globe_registration_output_dir_for_patient(patient_id, self.mri_series_description) / "globe_manual_registration_qc.json"
            if qc_path.exists():
                completed += 1
        return completed, len(patients)

    def registration_progress_text(self) -> str:
        completed, total = self.registration_progress_counts()
        return f"done={completed}/{total}"

    def update_progress_label(self) -> None:
        if self.progress_label is None:
            return
        completed, total = self.registration_progress_counts()
        self.progress_label.setText(f"Index {self.queue_position_text()} | Done {completed}/{total}")

    @staticmethod
    def _read_work_queue(path: Path | None) -> list[dict[str, str]]:
        if not path or not path.exists():
            return []
        with path.open(newline="", encoding="utf-8-sig") as f:
            return [dict(row) for row in csv.DictReader(f)]


def run(
    manifest_csv: str | Path,
    patient_id: str,
    db_path: str | Path,
    ct_series_description: str = "AX",
    mri_series_description: str = DEFAULT_MRI_SERIES,
    work_queue_csv: str | Path | None = None,
    annotator_id: str = "default",
) -> None:
    app = DualViewWorkbench(
        manifest_csv=manifest_csv,
        patient_id=patient_id,
        db_path=db_path,
        ct_series_description=ct_series_description,
        mri_series_description=mri_series_description,
        work_queue_csv=work_queue_csv,
        annotator_id=annotator_id,
    )
    app.launch()
