"""Minimal strength sessions: nullable declarations and immutable local revisions."""

from __future__ import annotations

import copy
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from .annotations import next_sequence, revision_key
from .models import DataError, number, utc_timestamp

FIELDS = {
    "body_scope",
    "session_rpe",
    "sensor",
    "start_utc",
    "end_utc",
    "notes",
    "remembered_exercise_ids",
    "selected_set_ids",
    "linked_csv_session_id",
    "annotation_elapsed_s",
    "annotation_actions",
    "measurement_opt_in",
}


def session_identity(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"fit-[0-9a-f]{64}|manual-[0-9a-f]{32}", value
    ):
        raise DataError("Identità seduta minima non valida")
    return value


def declarations(values: dict) -> dict:
    if not isinstance(values, dict) or set(values) - FIELDS:
        raise DataError("Campi seduta minima non validi")
    result = {k: values.get(k) for k in sorted(FIELDS)}
    for key, options in (
        ("body_scope", {None, "upper", "lower", "full_body"}),
        ("sensor", {None, "wrist", "chest", "other"}),
    ):
        if (
            result[key] is not None
            and not isinstance(result[key], str)
            or result[key] not in options
        ):
            raise DataError("Tipo seduta o sensore non valido")
    for key, minimum, maximum in (
        ("session_rpe", 1, 10),
        ("annotation_elapsed_s", 0, 86400),
        ("annotation_actions", 0, 100000),
    ):
        value = result[key]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or number(value, minimum=minimum, maximum=maximum) is None
        ):
            raise DataError("Numero di seduta non valido")
    if (
        result["annotation_actions"] is not None
        and int(result["annotation_actions"]) != result["annotation_actions"]
    ):
        raise DataError("Numero azioni intero richiesto")
    opt = result["measurement_opt_in"]
    if opt is not None and not isinstance(opt, bool):
        raise DataError("Consenso misura locale non valido")
    result["measurement_opt_in"] = opt is True
    if not result["measurement_opt_in"] and any(
        result[k] is not None for k in ("annotation_elapsed_s", "annotation_actions")
    ):
        raise DataError("Misure UX facoltative: consenso locale richiesto")
    for key in ("notes", "linked_csv_session_id"):
        value = result[key]
        if value is not None and (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > (2000 if key == "notes" else 200)
            or any(ord(c) < 32 and c not in "\n\t" for c in value)
        ):
            raise DataError("Testo o collegamento non valido")
    start, end = result["start_utc"], result["end_utc"]
    if (start is None) != (end is None):
        raise DataError("Dichiarare entrambi gli orari oppure lasciarli ignoti")
    if start is not None:
        result["start_utc"], result["end_utc"] = utc_timestamp(start), utc_timestamp(end)
        if result["start_utc"] >= result["end_utc"]:
            raise DataError("La fine deve seguire l’inizio")
    for key in ("remembered_exercise_ids", "selected_set_ids"):
        value = result[key] if result[key] is not None else []
        if (
            not isinstance(value, list)
            or len(value) > 300
            or any(not isinstance(v, str) or not 1 <= len(v) <= 120 for v in value)
            or len(set(value)) != len(value)
        ):
            raise DataError("Riferimenti esercizi/serie non validi")
        result[key] = list(value)
    return result


def validate_summary(row: dict) -> None:
    """Validate exported derivatives before aggregating; markers are declarations."""
    if (
        not isinstance(row, dict)
        or row.get("artifact_type") != "strength_session_summary"
        or row.get("schema_version") != 1
    ):
        raise DataError("Schema seduta minima non valido")
    key = session_identity(row.get("session_id"))
    clean = declarations(row.get("declarations"))
    source = key[4:] if key.startswith("fit-") else None
    if row.get("source_ref") != source or type(row.get("synthetic")) is not bool:
        raise DataError("Provenienza seduta minima non coerente")
    for field in ("start_utc", "end_utc", "recorded_start_utc", "recorded_end_utc"):
        stamp = row.get(field)
        if stamp is not None and utc_timestamp(stamp) != stamp:
            raise DataError("Timestamp derivati richiesti in UTC")
    for a, b in (("start_utc", "end_utc"), ("recorded_start_utc", "recorded_end_utc")):
        if (
            (row.get(a) is None) != (row.get(b) is None)
            or row.get(a) is not None
            and row[a] >= row[b]
        ):
            raise DataError("Estremi seduta minima non coerenti")
    for field in ("elapsed_duration_s", "declared_duration_s", "hr_support_s"):
        value = row.get(field)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or number(value, minimum=0 if field == "hr_support_s" else 0.001) is None
        ):
            raise DataError("Durata/supporto seduta minima non valido")
    if source is None and any(
        row.get(f) is not None
        for f in ("elapsed_duration_s", "hr_support_s", "recorded_start_utc", "recorded_end_utc")
    ):
        raise DataError("Seduta manuale senza misure FIT")
    if row.get("declared_duration_s") != (
        (
            datetime.fromisoformat(clean["end_utc"]) - datetime.fromisoformat(clean["start_utc"])
        ).total_seconds()
        if clean["start_utc"]
        else None
    ):
        raise DataError("Durata dichiarata incoerente con gli estremi")
    refs = row.get("selected_sets")
    if not isinstance(refs, list) or len(refs) > 300 or any(not isinstance(r, dict) for r in refs):
        raise DataError("Serie selezionate non valide")
    seen = set()
    for ref in refs:
        slot = ref.get("snapshot")
        if (
            not isinstance(slot, dict)
            or slot.get("confirmed") is not True
            or slot.get("kind") != "work"
            or slot.get("provenance") != "user_confirmed"
        ):
            raise DataError("Snapshot serie deve essere dichiarato e confermato")
        id = ref.get("slot_id")
        if (
            not isinstance(id, str)
            or id in seen
            or id != slot.get("slot_id")
            or ref.get("source_ref") != source
            or source is None
        ):
            raise DataError("Riferimenti serie duplicati o non coerenti")
        seen.add(id)
        exercise = slot.get("exercise_snapshot")
        if (
            not isinstance(exercise, dict)
            or exercise.get("id") != slot.get("exercise_id")
            or exercise.get("load_mode") not in {"external", "bodyweight"}
        ):
            raise DataError("Esercizio dello snapshot non valido")
        reps = slot.get("reps")
        if (
            isinstance(reps, bool)
            or not isinstance(reps, (int, float))
            or number(reps, minimum=1) is None
            or int(reps) != reps
        ):
            raise DataError("Ripetizioni dichiarate intere richieste")
        required = (
            {"external_load_kg"} if exercise["load_mode"] == "external" else {"added_load_kg"}
        )
        for field in ("external_load_kg", "added_load_kg", "body_mass_kg", "rpe", "rir"):
            value = slot.get(field)
            if (
                value is None
                and field in required
                or value is not None
                and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or number(
                        value,
                        minimum=1 if field in {"body_mass_kg", "rpe"} else 0,
                        maximum=10 if field in {"rpe", "rir"} else float("inf"),
                    )
                    is None
                )
            ):
                raise DataError("Carico/RPE/RIR dichiarato non valido")
    if seen != set(clean["selected_set_ids"]):
        raise DataError("Selezione serie incoerente con gli snapshot")


class StrengthSessions:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.root = self.workspace / "data" / "processed" / "hta-local" / "strength-sessions"

    @property
    def fit(self):
        # Import lazily: the FIT catalog retains the legacy LocalCatalog dependency.
        from .strength_catalog import StrengthCatalog

        return StrengthCatalog(self.workspace)

    def latest(self, session_id: str) -> dict | None:
        folder = self.root / session_identity(session_id) / "revisions"
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json")]
        return max(rows, key=revision_key) if rows else None

    def view(self, session_id: str) -> dict:
        row = self.latest(session_id)
        if row is not None:
            return row
        if not session_id.startswith("fit-"):
            raise DataError("Seduta manuale sconosciuta")
        return self._build(session_id, {}, synthetic=False)

    def _build(self, session_id: str, values: dict, *, synthetic: bool) -> dict:
        clean = declarations(values)
        context = self.fit.context()
        ids = {e["id"] for e in context["exercises"]}
        if any(i not in ids for i in clean["remembered_exercise_ids"]):
            raise DataError("Esercizio ricordato fuori catalogo")
        if clean["linked_csv_session_id"] is not None:
            from .local_catalog import LocalCatalog

            if clean["linked_csv_session_id"] not in {
                r.session_id for r in LocalCatalog(self.workspace).strength_sets()
            }:
                raise DataError("Sessione CSV da collegare non presente")
            if clean["start_utc"] is not None and any(
                not clean["start_utc"] <= r.timestamp_utc <= clean["end_utc"]
                for r in LocalCatalog(self.workspace).strength_sets()
                if r.session_id == clean["linked_csv_session_id"]
            ):
                raise DataError("Serie CSV fuori dagli orari dichiarati")
        source = session_id[4:] if session_id.startswith("fit-") else None
        raw = self.fit.raw(source) if source else None
        quality = self.fit.quality(source) if source else None
        draft = self.fit.latest(source) if source else None
        snapshots = []
        for key in clean["selected_set_ids"]:
            slot = next((s for s in (draft or {}).get("slots", []) if s["slot_id"] == key), None)
            if (
                slot is None
                or not slot["confirmed"]
                or slot["kind"] != "work"
                or slot["provenance"] != "user_confirmed"
            ):
                raise DataError("Selezionare soltanto serie realmente confermate")
            snapshots.append(
                {
                    "source_ref": source,
                    "revision_id": draft["revision_id"],
                    "slot_id": key,
                    "snapshot": copy.deepcopy(slot),
                }
            )
        recorded_start = raw["start_utc"] if raw else None
        elapsed = raw["duration_s"] if raw else None
        recorded_end = (
            (datetime.fromisoformat(recorded_start) + timedelta(seconds=elapsed)).isoformat()
            if raw
            else None
        )
        declared_duration = (
            (
                datetime.fromisoformat(clean["end_utc"])
                - datetime.fromisoformat(clean["start_utc"])
            ).total_seconds()
            if clean["start_utc"]
            else None
        )
        if clean["start_utc"] is not None:
            for ref in snapshots:
                slot = ref["snapshot"]
                if slot["start_s"] is not None:
                    a = (
                        datetime.fromisoformat(recorded_start) + timedelta(seconds=slot["start_s"])
                    ).isoformat()
                    b = (
                        datetime.fromisoformat(recorded_start) + timedelta(seconds=slot["end_s"])
                    ).isoformat()
                    if not clean["start_utc"] <= a < b <= clean["end_utc"]:
                        raise DataError("Serie confermata fuori dagli orari dichiarati")
        start = clean["start_utc"] or recorded_start
        end = clean["end_utc"] or recorded_end
        return {
            "artifact_type": "strength_session_summary",
            "schema_version": 1,
            "session_id": session_id,
            "source_ref": source,
            "synthetic": raw["synthetic"] if raw else synthetic,
            "start_utc": start,
            "end_utc": end,
            "recorded_start_utc": recorded_start,
            "recorded_end_utc": recorded_end,
            "elapsed_duration_s": elapsed,
            "declared_duration_s": declared_duration,
            "sensor": clean["sensor"],
            "declarations": clean,
            "remembered_exercises": [
                copy.deepcopy(e)
                for e in context["exercises"]
                if e["id"] in clean["remembered_exercise_ids"]
            ],
            "selected_sets": snapshots,
            "quality_ref": quality["quality_id"] if quality else None,
            "hr_support_s": quality["hr"]["supported_duration_s"] if quality else None,
            "review_state": "unreviewed",
            "detail_level": "selected_sets" if snapshots else "summary",
            "revision_id": None,
            "parent_revision_id": None,
            "created_utc": None,
            "provenance": {
                "start_utc": "user_declared"
                if clean["start_utc"]
                else quality["time_provenance"]["start_utc"]
                if quality
                else "unavailable",
                "end_utc": "user_declared"
                if clean["end_utc"]
                else "inferred_recorded_bounds"
                if raw
                else "unavailable",
                "elapsed_duration_s": quality["time_provenance"]["duration_s"]
                if quality
                else "unavailable",
                "declared_duration_s": "inferred_user_declared_bounds"
                if declared_duration is not None
                else "unavailable",
                "declarations": "user_declared",
                "selected_sets": "user_confirmed_snapshot",
                "hr_support_s": "inferred_valid_record_support" if quality else "unavailable",
                "ui_measurements": "observed_local_ui_opt_in"
                if clean["measurement_opt_in"]
                else "unavailable",
            },
            "limits": [
                "Elapsed clock duration is not muscle work duration",
                "Unknown sets/volume remain unknown; no fake CSV row",
                "Acknowledged does not confirm unknown exercises or device hints",
            ],
        }

    def save(
        self, session_id: str | None, parent: str | None, values: dict, *, synthetic: bool = False
    ) -> dict:
        if not isinstance(synthetic, bool):
            raise DataError("Flag sintetico non valido")
        if session_id is None:
            if parent is not None:
                raise DataError("Nuova seduta senza revisione precedente")
            session_id = "manual-" + uuid4().hex
        session_identity(session_id)
        previous = self.latest(session_id)
        if (previous["revision_id"] if previous else None) != parent:
            raise DataError("Revisione cambiata: ricarica; nessun dato sovrascritto")
        row = self._build(
            session_id, values, synthetic=previous["synthetic"] if previous else synthetic
        )
        link = row["declarations"]["linked_csv_session_id"] or (
            session_id if session_id.startswith("fit-") else None
        )
        if link and any(
            r["session_id"] != session_id
            and (r["declarations"]["linked_csv_session_id"] == link or r["session_id"] == link)
            for r in self.sessions()
        ):
            raise DataError("Sessione CSV già collegata a un’altra seduta")
        row.update(
            review_state="acknowledged",
            revision_sequence=next_sequence(self.root / session_id / "revisions"),
            parent_revision_id=parent,
            revision_id=uuid4().hex,
            created_utc=datetime.now(UTC).isoformat(),
        )
        folder = self.root / session_id / "revisions"
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / (row["revision_id"] + ".json")).open("x", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, allow_nan=False, indent=2)
        return row

    def sessions(self) -> list[dict]:
        result = [self.latest(p.name) for p in self.root.glob("*") if p.is_dir()]
        return sorted(
            (r for r in result if r),
            key=lambda r: (r["start_utc"] or "", r["session_id"]),
            reverse=True,
        )

    def export(self, session_id: str) -> bytes:
        row = self.latest(session_id)
        if row is None:
            raise DataError("Salva la seduta minima prima di esportare")
        return json.dumps(row, ensure_ascii=False, allow_nan=False, indent=2).encode()

    def ux_metrics(self) -> dict:
        rows = self.sessions()
        measured = [r["declarations"] for r in rows if r["declarations"]["measurement_opt_in"]]
        times = [
            r["annotation_elapsed_s"] for r in measured if r["annotation_elapsed_s"] is not None
        ]
        actions = [r["annotation_actions"] for r in measured if r["annotation_actions"] is not None]
        return {
            "saved_sessions": len(rows),
            "partial_sessions": sum(not r["selected_sets"] for r in rows),
            "opt_in_sessions": len(measured),
            "annotation_elapsed_observations": len(times),
            "annotation_elapsed_mean_s": sum(times) / len(times) if times else None,
            "annotation_actions_observations": len(actions),
            "annotation_actions_mean": sum(actions) / len(actions) if actions else None,
            "provenance": "optional_local_ui_measurements_latest_revision_only",
            "limits": [
                "No 30-second objective measured without observations",
                "No selected sets means no known detailed series, not no work",
            ],
        }
