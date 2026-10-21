"""Validated user declarations, separate from recorded channels; immutable revisions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .models import DataError, number, utc_timestamp

DEFAULTS = {
    "run_type": None,
    "ventilation": None,
    "sensor": None,
    "session_rpe": None,
    "incline_pct": None,
    "speed_confirmed": False,
    "constant_incline": False,
    "target_distance_m": None,
    "distance_source": None,
    "lower_body_exposure": None,
    "technique": None,
    "notes": None,
    "start_utc": None,
    "end_utc": None,
}


def validate_annotations(payload: dict) -> dict:
    if not isinstance(payload, dict) or set(payload) - DEFAULTS.keys():
        raise DataError("Annotazioni: campi sconosciuti o oggetto non valido")
    result = {**DEFAULTS, **payload}
    for field in ("speed_confirmed", "constant_incline"):
        if not isinstance(result[field], bool):
            raise DataError(f"{field}: booleano richiesto")
    for field, allowed in {
        "run_type": {"easy", "quality", "other"},
        "sensor": {"wrist", "chest", "other"},
        "distance_source": {"watch_final", "treadmill_log", "user_other"},
        "lower_body_exposure": {"none", "low", "moderate", "high"},
    }.items():
        value = result[field]
        if value is not None and (not isinstance(value, str) or value not in allowed):
            raise DataError(f"{field}: dichiarazione non valida")
    for field, lo, hi in (
        ("session_rpe", 1, 10),
        ("incline_pct", -20, 20),
        ("target_distance_m", 0.001, 1000000),
    ):
        value = result[field]
        if value is not None:
            clean = number(value, minimum=lo, maximum=hi)
            if clean is None:
                raise DataError(f"{field}: fuori range tecnico")
            result[field] = clean
    if (result["target_distance_m"] is None) != (result["distance_source"] is None):
        raise DataError("Distanza e fonte devono essere dichiarate insieme")
    for field in ("technique", "notes", "ventilation"):
        value = result[field]
        if value is not None and (not isinstance(value, str) or len(value) > 2000):
            raise DataError(f"{field}: testo massimo 2000 caratteri")
    for field in ("start_utc", "end_utc"):
        if result[field] is not None:
            result[field] = utc_timestamp(result[field])
    if (result["start_utc"] is None) != (result["end_utc"] is None):
        raise DataError("Inizio e fine dichiarati insieme")
    if result["start_utc"] and result["end_utc"] <= result["start_utc"]:
        raise DataError("Fine deve seguire inizio")
    return result


def revision_key(row: dict) -> tuple:
    # Additive order; legacy rows retain their previous fallback. Never forge UTC.
    return (row.get("revision_sequence", 0), row["created_utc"], row["revision_id"])


def next_sequence(directory: Path) -> int:
    # One writer per catalog (UI lock and workspace coordination); no process lock.
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in directory.glob("*.json")]
    return max((r.get("revision_sequence", 0) for r in rows), default=0) + 1


def append_revision(directory: Path, subject: str, payload: dict) -> dict:
    values = validate_annotations(payload)
    if not subject or len(subject) > 128:
        raise DataError("Identificatore annotazione non valido")
    revision = {
        "schema_version": 1,
        "artifact_type": "user_annotations",
        "revision_sequence": next_sequence(directory),
        "revision_id": uuid4().hex,
        "subject": subject,
        "created_utc": datetime.now(UTC).isoformat(),
        "provenance": "user_declared",
        "values": values,
    }
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / (revision["revision_id"] + ".json")).open("x", encoding="utf-8") as out:
        json.dump(revision, out, ensure_ascii=False, allow_nan=False, indent=2)
    return revision
