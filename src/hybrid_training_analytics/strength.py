"""Editable strength CSV: one row per performed set, explicit load semantics."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .models import DataError, number, utc_timestamp

COLUMNS = [
    "timestamp",
    "session_id",
    "exercise",
    "load_mode",
    "body_mass_kg",
    "added_load_kg",
    "external_load_kg",
    "set_index",
    "reps",
    "rir",
    "rpe",
]


@dataclass(frozen=True)
class StrengthSet:
    timestamp_utc: str
    session_id: str
    exercise: str
    load_mode: str
    body_mass_kg: float
    added_load_kg: float
    external_load_kg: float
    set_index: int
    reps: int
    rir: float | None
    rpe: float | None


def read_strength(path: Path) -> list[StrengthSet]:
    result = []
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise DataError(f"Colonne diario richieste (in ordine): {','.join(COLUMNS)}")
        for index, row in enumerate(reader, 2):
            try:
                stamp = utc_timestamp(row["timestamp"])
                values = [
                    number(row[key])
                    for key in (
                        "body_mass_kg",
                        "added_load_kg",
                        "external_load_kg",
                        "set_index",
                        "reps",
                    )
                ]
                if any(value is None for value in values):
                    raise DataError("Carichi/ripetizioni non numerici, negativi o non finiti")
                body, added, external, set_index, reps = values
                if (
                    body <= 0
                    or set_index < 1
                    or reps < 1
                    or (int(set_index) != set_index or int(reps) != reps)
                ):
                    raise DataError("Massa >0; indice serie e ripetizioni interi >=1")
                mode = row["load_mode"]
                if mode not in {"bodyweight", "external"}:
                    raise DataError("load_mode deve essere bodyweight oppure external")
                if (mode == "bodyweight" and external != 0) or (mode == "external" and added != 0):
                    raise DataError("Carichi incoerenti con load_mode")
                rir = None if row["rir"] == "" else number(row["rir"], maximum=10)
                rpe = None if row["rpe"] == "" else number(row["rpe"], minimum=1, maximum=10)
                if (row["rir"] and rir is None) or (row["rpe"] and rpe is None):
                    raise DataError("RIR 0..10; RPE 1..10; vuoto se sconosciuto")
                session, exercise = row["session_id"].strip(), row["exercise"].strip()
                if not session or not exercise:
                    raise DataError("session_id/exercise mancanti")
                identity = session, exercise, int(set_index)
                if identity in seen:
                    raise DataError("Serie duplicata per sessione/esercizio/indice")
                seen.add(identity)
                result.append(
                    StrengthSet(
                        stamp,
                        session,
                        exercise,
                        mode,
                        body,
                        added,
                        external,
                        int(set_index),
                        int(reps),
                        rir,
                        rpe,
                    )
                )
            except (DataError, TypeError, KeyError) as exc:
                raise DataError(f"Diario riga {index}: {exc}") from exc
    if not result:
        raise DataError("Diario forza vuoto")
    return sorted(result, key=lambda s: (s.timestamp_utc, s.session_id, s.exercise, s.set_index))


def summarize_strength(sets: list[StrengthSet]) -> list[dict]:
    groups = defaultdict(list)
    for row in sets:
        groups[(row.session_id, row.exercise, row.load_mode)].append(row)
    result = []
    for (session, exercise, mode), rows in groups.items():
        best = max(rows, key=lambda s: (s.added_load_kg + s.external_load_kg, s.reps))
        result.append(
            {
                "timestamp_utc": min(row.timestamp_utc for row in rows),
                "session_id": session,
                "exercise": exercise,
                "load_mode": mode,
                "sets": len(rows),
                "reps": sum(row.reps for row in rows),
                "external_volume_kg_reps": sum(
                    (row.added_load_kg + row.external_load_kg) * row.reps for row in rows
                ),
                "system_mass_proxy_kg_reps": sum(
                    (row.body_mass_kg + row.added_load_kg) * row.reps for row in rows
                )
                if mode == "bodyweight"
                else None,
                "best_observed_load_kg": best.added_load_kg + best.external_load_kg,
                "best_observed_reps": best.reps,
                "best_observed_rir": best.rir,
                "body_mass_kg": rows[0].body_mass_kg,
            }
        )
    return sorted(result, key=lambda row: (row["timestamp_utc"], row["exercise"]))
