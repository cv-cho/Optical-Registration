from __future__ import annotations

import json
import platform
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

from . import __version__
from .schema import validate_landmark_label, validate_modality


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def current_software_versions() -> dict[str, str | None]:
    return {
        "annotation_app_version": __version__,
        "python_version": platform.python_version(),
        "simpleitk_version": package_version("SimpleITK"),
        "pydicom_version": package_version("pydicom"),
        "napari_version": package_version("napari"),
        "magicgui_version": package_version("magicgui"),
    }


@dataclass
class LandmarkRecord:
    patient_id: str
    study_uid: str
    series_uid: str
    modality: str
    landmark_label: str
    voxel_zyx: list[float]
    itk_index_xyz: list[float]
    physical_lps_mm: list[float]
    view_used: str
    slice_index_used: float | None
    image_spacing_xyz: list[float]
    image_origin_lps: list[float]
    image_direction_3x3: list[list[float]]
    source: str = "manual"
    visibility: str = "visible"
    use_for_transform: bool = True
    quality: int = 0
    annotator_id: str = "default"
    qc_status: str = "unchecked"
    created_at: str = ""
    updated_at: str = ""
    software_versions: dict[str, Any] | None = None

    def normalized(self) -> "LandmarkRecord":
        self.modality = validate_modality(self.modality)
        self.landmark_label = validate_landmark_label(self.landmark_label)
        if not self.created_at:
            self.created_at = utc_now()
        self.updated_at = utc_now()
        if self.software_versions is None:
            self.software_versions = current_software_versions()
        return self

    def to_jsonable(self) -> dict[str, Any]:
        item = asdict(self)
        item["use_for_transform"] = bool(item["use_for_transform"])
        return item


@dataclass
class GlobeSurfacePointRecord:
    patient_id: str
    study_uid: str
    series_uid: str
    modality: str
    side: str
    voxel_zyx: list[float]
    itk_index_xyz: list[float]
    physical_lps_mm: list[float]
    view_used: str
    slice_index_used: float | None
    image_spacing_xyz: list[float]
    image_origin_lps: list[float]
    image_direction_3x3: list[list[float]]
    source: str = "manual"
    annotator_id: str = "default"
    created_at: str = ""
    software_versions: dict[str, Any] | None = None

    def normalized(self) -> "GlobeSurfacePointRecord":
        self.modality = validate_modality(self.modality)
        self.side = self.side.upper()
        if self.side not in ("L", "R"):
            raise ValueError(f"Unknown globe side: {self.side}")
        if not self.created_at:
            self.created_at = utc_now()
        if self.software_versions is None:
            self.software_versions = current_software_versions()
        return self

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GlobeCenterOverrideRecord:
    patient_id: str
    study_uid: str
    series_uid: str
    modality: str
    side: str
    physical_lps_mm: list[float]
    source: str = "manual_globe_center_override"
    annotator_id: str = "default"
    updated_at: str = ""
    software_versions: dict[str, Any] | None = None

    def normalized(self) -> "GlobeCenterOverrideRecord":
        self.modality = validate_modality(self.modality)
        self.side = self.side.upper()
        if self.side not in ("L", "R"):
            raise ValueError(f"Unknown globe side: {self.side}")
        self.updated_at = utc_now()
        if self.software_versions is None:
            self.software_versions = current_software_versions()
        return self

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS landmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    study_uid TEXT NOT NULL,
    series_uid TEXT NOT NULL,
    modality TEXT NOT NULL,
    landmark_label TEXT NOT NULL,
    voxel_zyx TEXT NOT NULL,
    itk_index_xyz TEXT NOT NULL,
    physical_lps_mm TEXT NOT NULL,
    view_used TEXT NOT NULL,
    slice_index_used REAL,
    image_spacing_xyz TEXT NOT NULL,
    image_origin_lps TEXT NOT NULL,
    image_direction_3x3 TEXT NOT NULL,
    source TEXT NOT NULL,
    visibility TEXT NOT NULL,
    use_for_transform INTEGER NOT NULL,
    quality INTEGER NOT NULL,
    annotator_id TEXT NOT NULL,
    qc_status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    software_versions TEXT NOT NULL,
    UNIQUE(patient_id, modality, series_uid, landmark_label)
);

CREATE TABLE IF NOT EXISTS annotation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    modality TEXT NOT NULL,
    landmark_label TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS globe_surface_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    study_uid TEXT NOT NULL,
    series_uid TEXT NOT NULL,
    modality TEXT NOT NULL,
    side TEXT NOT NULL,
    voxel_zyx TEXT NOT NULL,
    itk_index_xyz TEXT NOT NULL,
    physical_lps_mm TEXT NOT NULL,
    view_used TEXT NOT NULL,
    slice_index_used REAL,
    image_spacing_xyz TEXT NOT NULL,
    image_origin_lps TEXT NOT NULL,
    image_direction_3x3 TEXT NOT NULL,
    source TEXT NOT NULL,
    annotator_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    software_versions TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS globe_manual_parameters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    ct_series_uid TEXT NOT NULL,
    mri_series_uid TEXT NOT NULL,
    mri_series_description TEXT NOT NULL,
    pitch_deg REAL NOT NULL,
    scale_xyz TEXT NOT NULL,
    annotator_id TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    software_versions TEXT NOT NULL,
    UNIQUE(patient_id, ct_series_uid, mri_series_uid)
);

CREATE TABLE IF NOT EXISTS globe_center_overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    study_uid TEXT NOT NULL,
    series_uid TEXT NOT NULL,
    modality TEXT NOT NULL,
    side TEXT NOT NULL,
    physical_lps_mm TEXT NOT NULL,
    source TEXT NOT NULL,
    annotator_id TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    software_versions TEXT NOT NULL,
    UNIQUE(patient_id, modality, series_uid, side)
);
"""


JSON_COLUMNS = {
    "voxel_zyx",
    "itk_index_xyz",
    "physical_lps_mm",
    "image_spacing_xyz",
    "image_origin_lps",
    "image_direction_3x3",
    "scale_xyz",
    "software_versions",
}


class AnnotationStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        self._migrate_landmarks_unique_key()
        self.conn.commit()

    def upsert_landmark(self, record: LandmarkRecord, event_type: str = "upsert") -> None:
        record = record.normalized()
        payload = record.to_jsonable()
        row = self._serialize_row(payload)
        columns = list(row)
        placeholders = ", ".join("?" for _ in columns)
        update_columns = [col for col in columns if col != "created_at"]
        updates = ", ".join(f"{col}=excluded.{col}" for col in update_columns)
        sql = (
            f"INSERT INTO landmarks ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(patient_id, modality, series_uid, landmark_label) DO UPDATE SET {updates}"
        )
        self.conn.execute(sql, [row[col] for col in columns])
        self.conn.execute(
            """
            INSERT INTO annotation_events
                (patient_id, modality, landmark_label, event_type, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.patient_id,
                record.modality,
                record.landmark_label,
                event_type,
                json.dumps(payload, ensure_ascii=True),
                utc_now(),
            ),
        )
        self.conn.commit()

    def delete_landmark(
        self,
        patient_id: str,
        modality: str,
        landmark_label: str,
        series_uid: str | None = None,
    ) -> None:
        modality = validate_modality(modality)
        landmark_label = validate_landmark_label(landmark_label)
        params = [str(patient_id), modality, landmark_label]
        sql = "DELETE FROM landmarks WHERE patient_id=? AND modality=? AND landmark_label=?"
        if series_uid:
            sql += " AND series_uid=?"
            params.append(str(series_uid))
        self.conn.execute(sql, params)
        payload = {"series_uid": str(series_uid)} if series_uid else {}
        self.conn.execute(
            """
            INSERT INTO annotation_events
                (patient_id, modality, landmark_label, event_type, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (str(patient_id), modality, landmark_label, "delete", json.dumps(payload, ensure_ascii=True), utc_now()),
        )
        self.conn.commit()

    def fetch_landmarks(
        self,
        patient_id: str,
        modality: str | None = None,
        series_uid: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["patient_id=?"]
        params = [str(patient_id)]
        if modality:
            clauses.append("modality=?")
            params.append(validate_modality(modality))
        if series_uid:
            clauses.append("series_uid=?")
            params.append(str(series_uid))
        sql = (
            "SELECT * FROM landmarks WHERE "
            + " AND ".join(clauses)
            + " ORDER BY modality, series_uid, landmark_label"
        )
        rows = self.conn.execute(sql, params).fetchall()
        return [self._deserialize_row(dict(row)) for row in rows]

    def add_globe_surface_point(self, record: GlobeSurfacePointRecord) -> int:
        record = record.normalized()
        payload = record.to_jsonable()
        row = self._serialize_json_row(payload)
        columns = list(row)
        placeholders = ", ".join("?" for _ in columns)
        cursor = self.conn.execute(
            f"INSERT INTO globe_surface_points ({', '.join(columns)}) VALUES ({placeholders})",
            [row[col] for col in columns],
        )
        self.conn.execute(
            """
            INSERT INTO annotation_events
                (patient_id, modality, landmark_label, event_type, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.patient_id,
                record.modality,
                f"{record.side}_GLOBE_SURFACE_POINT",
                "globe_surface_add",
                json.dumps(payload, ensure_ascii=True),
                utc_now(),
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def fetch_globe_surface_points(
        self,
        patient_id: str,
        modality: str | None = None,
        series_uid: str | None = None,
        side: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["patient_id=?"]
        params = [str(patient_id)]
        if modality:
            clauses.append("modality=?")
            params.append(validate_modality(modality))
        if series_uid:
            clauses.append("series_uid=?")
            params.append(str(series_uid))
        if side:
            side_value = side.upper()
            if side_value not in ("L", "R"):
                raise ValueError(f"Unknown globe side: {side}")
            clauses.append("side=?")
            params.append(side_value)
        sql = (
            "SELECT * FROM globe_surface_points WHERE "
            + " AND ".join(clauses)
            + " ORDER BY modality, series_uid, side, id"
        )
        rows = self.conn.execute(sql, params).fetchall()
        return [self._deserialize_row(dict(row), keep_id=True) for row in rows]

    def upsert_globe_center_override(self, record: GlobeCenterOverrideRecord) -> None:
        record = record.normalized()
        payload = record.to_jsonable()
        row = self._serialize_json_row(payload)
        columns = list(row)
        placeholders = ", ".join("?" for _ in columns)
        update_columns = [col for col in columns if col not in {"patient_id", "modality", "series_uid", "side"}]
        updates = ", ".join(f"{col}=excluded.{col}" for col in update_columns)
        sql = (
            f"INSERT INTO globe_center_overrides ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(patient_id, modality, series_uid, side) DO UPDATE SET {updates}"
        )
        self.conn.execute(sql, [row[col] for col in columns])
        self.conn.execute(
            """
            INSERT INTO annotation_events
                (patient_id, modality, landmark_label, event_type, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.patient_id,
                record.modality,
                f"{record.side}_GLOBE_CENTER_OVERRIDE",
                "globe_center_override_upsert",
                json.dumps(payload, ensure_ascii=True),
                utc_now(),
            ),
        )
        self.conn.commit()

    def fetch_globe_center_overrides(
        self,
        patient_id: str,
        modality: str | None = None,
        series_uid: str | None = None,
        side: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["patient_id=?"]
        params = [str(patient_id)]
        if modality:
            clauses.append("modality=?")
            params.append(validate_modality(modality))
        if series_uid:
            clauses.append("series_uid=?")
            params.append(str(series_uid))
        if side:
            side_value = side.upper()
            if side_value not in ("L", "R"):
                raise ValueError(f"Unknown globe side: {side}")
            clauses.append("side=?")
            params.append(side_value)
        sql = (
            "SELECT * FROM globe_center_overrides WHERE "
            + " AND ".join(clauses)
            + " ORDER BY modality, series_uid, side"
        )
        rows = self.conn.execute(sql, params).fetchall()
        return [self._deserialize_row(dict(row), keep_id=True) for row in rows]

    def delete_globe_center_override(
        self,
        patient_id: str,
        modality: str,
        series_uid: str,
        side: str,
    ) -> bool:
        modality_value = validate_modality(modality)
        side_value = side.upper()
        if side_value not in ("L", "R"):
            raise ValueError(f"Unknown globe side: {side}")
        cursor = self.conn.execute(
            """
            DELETE FROM globe_center_overrides
            WHERE patient_id=? AND modality=? AND series_uid=? AND side=?
            """,
            (str(patient_id), modality_value, str(series_uid), side_value),
        )
        deleted = int(cursor.rowcount or 0) > 0
        self.conn.execute(
            """
            INSERT INTO annotation_events
                (patient_id, modality, landmark_label, event_type, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(patient_id),
                modality_value,
                f"{side_value}_GLOBE_CENTER_OVERRIDE",
                "globe_center_override_delete",
                json.dumps({"series_uid": str(series_uid), "deleted": deleted}, ensure_ascii=True),
                utc_now(),
            ),
        )
        self.conn.commit()
        return deleted

    def delete_globe_surface_point(self, point_id: int) -> None:
        row = self.conn.execute("SELECT * FROM globe_surface_points WHERE id=?", (int(point_id),)).fetchone()
        if row is None:
            return
        item = self._deserialize_row(dict(row), keep_id=True)
        self.conn.execute("DELETE FROM globe_surface_points WHERE id=?", (int(point_id),))
        self.conn.execute(
            """
            INSERT INTO annotation_events
                (patient_id, modality, landmark_label, event_type, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                item["patient_id"],
                item["modality"],
                f"{item['side']}_GLOBE_SURFACE_POINT",
                "globe_surface_delete",
                json.dumps({"id": int(point_id)}, ensure_ascii=True),
                utc_now(),
            ),
        )
        self.conn.commit()

    def update_globe_surface_point(
        self,
        point_id: int,
        record: GlobeSurfacePointRecord,
        event_type: str = "globe_surface_update",
    ) -> None:
        previous = self.conn.execute("SELECT * FROM globe_surface_points WHERE id=?", (int(point_id),)).fetchone()
        if previous is None:
            return
        previous_item = self._deserialize_row(dict(previous), keep_id=True)
        if not record.created_at:
            record.created_at = str(previous_item.get("created_at", ""))
        record = record.normalized()
        payload = record.to_jsonable()
        row = self._serialize_json_row(payload)
        assignments = ", ".join(f"{column}=?" for column in row)
        self.conn.execute(
            f"UPDATE globe_surface_points SET {assignments} WHERE id=?",
            [row[column] for column in row] + [int(point_id)],
        )
        self.conn.execute(
            """
            INSERT INTO annotation_events
                (patient_id, modality, landmark_label, event_type, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.patient_id,
                record.modality,
                f"{record.side}_GLOBE_SURFACE_POINT",
                event_type,
                json.dumps({"id": int(point_id), **payload}, ensure_ascii=True),
                utc_now(),
            ),
        )
        self.conn.commit()

    def upsert_globe_manual_parameters(
        self,
        patient_id: str,
        ct_series_uid: str,
        mri_series_uid: str,
        mri_series_description: str,
        pitch_deg: float,
        scale_xyz: tuple[float, float, float],
        annotator_id: str = "default",
    ) -> None:
        payload = {
            "patient_id": str(patient_id),
            "ct_series_uid": str(ct_series_uid),
            "mri_series_uid": str(mri_series_uid),
            "mri_series_description": str(mri_series_description),
            "pitch_deg": float(pitch_deg),
            "scale_xyz": [float(v) for v in scale_xyz],
            "annotator_id": str(annotator_id),
            "updated_at": utc_now(),
            "software_versions": current_software_versions(),
        }
        row = self._serialize_json_row(payload)
        columns = list(row)
        placeholders = ", ".join("?" for _ in columns)
        update_columns = [col for col in columns if col not in {"patient_id", "ct_series_uid", "mri_series_uid"}]
        updates = ", ".join(f"{col}=excluded.{col}" for col in update_columns)
        sql = (
            f"INSERT INTO globe_manual_parameters ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(patient_id, ct_series_uid, mri_series_uid) DO UPDATE SET {updates}"
        )
        self.conn.execute(sql, [row[col] for col in columns])
        self.conn.commit()

    def fetch_globe_manual_parameters(
        self,
        patient_id: str,
        ct_series_uid: str,
        mri_series_uid: str,
    ) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT * FROM globe_manual_parameters
            WHERE patient_id=? AND ct_series_uid=? AND mri_series_uid=?
            """,
            (str(patient_id), str(ct_series_uid), str(mri_series_uid)),
        ).fetchone()
        if row is None:
            return None
        return self._deserialize_row(dict(row))

    def count_patient_data(self, patient_id: str) -> dict[str, int]:
        patient_id = str(patient_id)
        counts: dict[str, int] = {}
        for table in [
            "landmarks",
            "annotation_events",
            "globe_surface_points",
            "globe_manual_parameters",
            "globe_center_overrides",
        ]:
            if table not in self._table_names():
                counts[table] = 0
                continue
            counts[table] = int(
                self.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE patient_id=?", (patient_id,)).fetchone()[0]
            )
        return counts

    def delete_patient_data(self, patient_id: str) -> dict[str, int]:
        patient_id = str(patient_id)
        counts = self.count_patient_data(patient_id)
        with self.conn:
            for table in [
                "landmarks",
                "annotation_events",
                "globe_surface_points",
                "globe_manual_parameters",
                "globe_center_overrides",
            ]:
                if table in self._table_names():
                    self.conn.execute(f"DELETE FROM {table} WHERE patient_id=?", (patient_id,))
        return counts

    def export_jsonl(self, out_path: str | Path, patient_ids: Iterable[str] | None = None) -> None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if patient_ids is None:
            rows = self.conn.execute("SELECT * FROM landmarks ORDER BY patient_id, modality, landmark_label").fetchall()
        else:
            ids = [str(v) for v in patient_ids]
            placeholders = ", ".join("?" for _ in ids)
            rows = self.conn.execute(
                f"SELECT * FROM landmarks WHERE patient_id IN ({placeholders}) ORDER BY patient_id, modality, landmark_label",
                ids,
            ).fetchall()
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(self._deserialize_row(dict(row)), ensure_ascii=True) + "\n")

    def _migrate_landmarks_unique_key(self) -> None:
        if self._has_series_uid_unique_key():
            return
        if not self._landmarks_table_exists():
            return
        backup_path = self.db_path.with_suffix(self.db_path.suffix + ".pre_series_uid_unique.bak")
        if self.db_path.exists() and not backup_path.exists():
            shutil.copyfile(self.db_path, backup_path)
        legacy_table = "landmarks_pre_series_uid_unique"
        suffix = 1
        existing_tables = self._table_names()
        while legacy_table in existing_tables:
            suffix += 1
            legacy_table = f"landmarks_pre_series_uid_unique_{suffix}"
        columns = [
            "id",
            "patient_id",
            "study_uid",
            "series_uid",
            "modality",
            "landmark_label",
            "voxel_zyx",
            "itk_index_xyz",
            "physical_lps_mm",
            "view_used",
            "slice_index_used",
            "image_spacing_xyz",
            "image_origin_lps",
            "image_direction_3x3",
            "source",
            "visibility",
            "use_for_transform",
            "quality",
            "annotator_id",
            "qc_status",
            "created_at",
            "updated_at",
            "software_versions",
        ]
        self.conn.execute(f"ALTER TABLE landmarks RENAME TO {legacy_table}")
        self.conn.executescript(SCHEMA_SQL)
        joined = ", ".join(columns)
        self.conn.execute(f"INSERT INTO landmarks ({joined}) SELECT {joined} FROM {legacy_table}")
        self.conn.execute(f"DROP TABLE {legacy_table}")

    def _has_series_uid_unique_key(self) -> bool:
        if not self._landmarks_table_exists():
            return False
        expected = ["patient_id", "modality", "series_uid", "landmark_label"]
        for index in self.conn.execute("PRAGMA index_list(landmarks)").fetchall():
            if not int(index["unique"]):
                continue
            rows = self.conn.execute(f"PRAGMA index_info({index['name']})").fetchall()
            columns = [str(row["name"]) for row in rows]
            if columns == expected:
                return True
        return False

    def _landmarks_table_exists(self) -> bool:
        return "landmarks" in self._table_names()

    def _table_names(self) -> set[str]:
        rows = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return {str(row["name"]) for row in rows}

    @staticmethod
    def _serialize_row(payload: dict[str, Any]) -> dict[str, Any]:
        row = dict(payload)
        row["use_for_transform"] = 1 if row["use_for_transform"] else 0
        return AnnotationStore._serialize_json_row(row)

    @staticmethod
    def _serialize_json_row(payload: dict[str, Any]) -> dict[str, Any]:
        row = dict(payload)
        for column in JSON_COLUMNS:
            if column in row:
                row[column] = json.dumps(row[column], ensure_ascii=True)
        return row

    @staticmethod
    def _deserialize_row(row: dict[str, Any], keep_id: bool = False) -> dict[str, Any]:
        if not keep_id:
            row.pop("id", None)
        if "use_for_transform" in row:
            row["use_for_transform"] = bool(row["use_for_transform"])
        for column in JSON_COLUMNS:
            if column in row:
                row[column] = json.loads(row[column])
        return row
