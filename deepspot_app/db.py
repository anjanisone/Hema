"""
SQLite database for DeepSpot. Supports network drive paths.
Schema: runs, images, metrics; app config.
"""
import sqlite3
import os
from datetime import datetime
from .config import get_db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    folder_path TEXT,
    crop_type TEXT,
    model_name TEXT,
    objective TEXT,
    num_images INTEGER,
    status TEXT
);

CREATE TABLE IF NOT EXISTS image_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    image_path TEXT NOT NULL,
    image_name TEXT NOT NULL,
    total_detections INTEGER NOT NULL,
    viable_count INTEGER NOT NULL,
    viability_pct REAL,
    inference_time_ms REAL,
    avg_diameter_um REAL,
    avg_area_um2 REAL,
    avg_perimeter_um REAL,
    avg_circularity REAL,
    avg_aspect_ratio REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_config (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_image_results_run ON image_results(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
"""

def connect():
    path = get_db_path()
    # Ensure parent dir exists (e.g. on network share)
    parent = os.path.dirname(path)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError:
            pass
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with connect() as c:
        c.executescript(SCHEMA)


def clear_all():
    """Remove all data (runs, image_results, app_config) so the DB can be shared without exposing paths."""
    with connect() as conn:
        conn.execute("DELETE FROM image_results")
        conn.execute("DELETE FROM runs")
        conn.execute("DELETE FROM app_config")
        conn.commit()

def insert_run(folder_path, crop_type, model_name, objective, num_images, status="completed"):
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO runs (created_at, folder_path, crop_type, model_name, objective, num_images, status)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (datetime.utcnow().isoformat(), folder_path, crop_type, model_name, objective, num_images, status),
        )
        conn.commit()
        return cur.lastrowid

def insert_image_result(run_id, image_path, image_name, total_detections, viable_count,
                        viability_pct, inference_time_ms, avg_diameter_um, avg_area_um2,
                        avg_perimeter_um, avg_circularity, avg_aspect_ratio):
    with connect() as c:
        c.execute(
            """INSERT INTO image_results
               (run_id, image_path, image_name, total_detections, viable_count, viability_pct,
                inference_time_ms, avg_diameter_um, avg_area_um2, avg_perimeter_um,
                avg_circularity, avg_aspect_ratio, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, image_path, image_name, total_detections, viable_count, viability_pct,
                inference_time_ms, avg_diameter_um, avg_area_um2, avg_perimeter_um,
                avg_circularity, avg_aspect_ratio, datetime.utcnow().isoformat(),
            ),
        )

def get_recent_runs(limit=100):
    with connect() as conn:
        cur = conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]

def get_image_results_for_run(run_id):
    """All image-level results for a run (for history detail)."""
    with connect() as conn:
        cur = conn.execute(
            """SELECT image_name, total_detections, viable_count, viability_pct,
                      inference_time_ms, avg_diameter_um, avg_area_um2, avg_circularity, avg_aspect_ratio
               FROM image_results WHERE run_id = ? ORDER BY id""",
            (run_id,),
        )
        return [dict(r) for r in cur.fetchall()]

def get_recent_image_results(limit=200):
    """Recent image-level results with folder path (for History list). Join with runs to get folder_path."""
    with connect() as conn:
        cur = conn.execute(
            """SELECT ir.id, ir.run_id, ir.image_path, ir.image_name, ir.total_detections, ir.viable_count,
                      ir.viability_pct, ir.inference_time_ms, ir.avg_diameter_um, ir.avg_area_um2,
                      ir.avg_perimeter_um, ir.avg_circularity, ir.avg_aspect_ratio, ir.created_at,
                      r.folder_path
               FROM image_results ir
               JOIN runs r ON r.id = ir.run_id
               ORDER BY ir.created_at DESC LIMIT ?""",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]

def set_config(key, value):
    with connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO app_config (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, datetime.utcnow().isoformat()),
        )

def get_config(key, default=None):
    with connect() as conn:
        cur = conn.execute("SELECT value FROM app_config WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else default
