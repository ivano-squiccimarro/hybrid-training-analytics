"""Private append-only strength reconciliation storage and explicit legacy CSV export."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from .annotations import next_sequence, revision_key
from .exercise_vocabulary import normalize_categories
from .local_catalog import identity
from .models import DataError
from .reconciliation import (
    DEFAULT_EXERCISES,
    apply_edit,
    detect_slots,
    suggest_exercises,
    validate_context,
)
from .reporting import write_json
from .strength import COLUMNS
from .strength_fit import read_strength_fit
from .strength_input import unpack_strength_input
from .strength_quality import assess_strength_quality


class StrengthCatalog:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.root = self.workspace / "data" / "processed" / "hta-local" / "strength-fit"

    def import_fit(self, content: bytes, *, synthetic: bool = False) -> str:
        content, descriptor = unpack_strength_input(content)
        key = hashlib.sha256(content).hexdigest()
        directory = self.root / key
        if directory.exists():
            self.raw(key)
            self._input_provenance(key, descriptor)
            self.quality(key)
            return key
        scratch = self.workspace / "work" / ("strength-import-" + uuid4().hex)
        scratch.mkdir(parents=True)
        path = scratch / "input.fit"
        path.write_bytes(content)
        raw = read_strength_fit(path, synthetic=synthetic)
        hypothesis = detect_slots(raw)
        directory.mkdir(parents=True, exist_ok=False)
        marker = directory / "INCOMPLETE.txt"
        marker.write_text("Importazione forza in corso", encoding="utf-8")
        (directory / "source.fit").write_bytes(content)
        write_json(directory / "raw.json", raw)
        write_json(directory / "hypotheses.json", hypothesis)
        self._save(key, hypothesis, None)
        marker.unlink()
        self._input_provenance(key, descriptor)
        self.quality(key)
        return key

    def raw(self, key: str) -> dict:
        directory = self.root / identity(key)
        if (directory / "INCOMPLETE.txt").exists():
            raise DataError("Import forza incompleto, conservato")
        return json.loads((directory / "raw.json").read_text(encoding="utf-8"))

    def _input_provenance(self, key: str, descriptor: dict) -> None:
        directory = self.root / identity(key) / "inputs"
        directory.mkdir(exist_ok=True)
        path = directory / (descriptor["input_sha256"] + ".json")
        if not path.exists():
            with path.open("x", encoding="utf-8") as out:
                json.dump(descriptor, out, ensure_ascii=False, allow_nan=False, indent=2)

    def quality(self, key: str) -> dict:
        raw = self.raw(key)
        result = assess_strength_quality(raw)
        result["raw_artifact_sha256"] = hashlib.sha256(
            (self.root / key / "raw.json").read_bytes()
        ).hexdigest()
        result["code_sha256"] = {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("strength_quality.py", "exercise_vocabulary.py")
        }
        content = json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True).encode()
        quality_id = hashlib.sha256(content).hexdigest()
        directory = self.root / key / "quality"
        directory.mkdir(exist_ok=True)
        path = directory / (quality_id + ".json")
        if not path.exists():
            with path.open("xb") as out:
                out.write(content)
        return {**result, "quality_id": quality_id}

    def export_quality(self, key: str) -> bytes:
        # Independent of detailed review/finalization; no invented series.
        return json.dumps(self.quality(key), ensure_ascii=False, allow_nan=False, indent=2).encode()

    def latest(self, key: str) -> dict:
        self.raw(key)
        rows = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in (self.root / key / "revisions").glob("*.json")
        ]
        if not rows:
            raise DataError("Revisione forza mancante")
        return max(rows, key=revision_key)

    def _save(self, key: str, draft: dict, parent: str | None) -> dict:
        payload = {
            **draft,
            "revision_id": uuid4().hex,
            "parent_revision_id": parent,
            "revision_sequence": next_sequence(self.root / key / "revisions"),
            "created_utc": datetime.now(UTC).isoformat(),
        }
        directory = self.root / key / "revisions"
        directory.mkdir(exist_ok=True)
        with (directory / (payload["revision_id"] + ".json")).open("x", encoding="utf-8") as out:
            json.dump(payload, out, ensure_ascii=False, allow_nan=False, indent=2)
        return payload

    def context(self) -> dict:
        rows = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in (self.root / "context").glob("*.json")
        ]
        values = (
            max(rows, key=revision_key)["values"]
            if rows
            else {"exercises": DEFAULT_EXERCISES, "active_plan": [], "history": []}
        )
        return validate_context(values)

    def save_context(self, values: dict) -> dict:
        clean = validate_context(values)
        directory = self.root / "context"
        directory.mkdir(parents=True, exist_ok=True)
        row = {
            "revision_id": uuid4().hex,
            "created_utc": datetime.now(UTC).isoformat(),
            "values": clean,
            "revision_sequence": next_sequence(directory),
            "provenance": "user_declared_catalog_plan_history",
        }
        with (directory / (row["revision_id"] + ".json")).open("x", encoding="utf-8") as out:
            json.dump(row, out, ensure_ascii=False, allow_nan=False, indent=2)
        return row

    def sessions(self) -> list[dict]:
        result = []
        for path in self.root.glob("*"):
            if not path.is_dir() or len(path.name) != 64 or (path / "INCOMPLETE.txt").exists():
                continue
            raw = self.raw(path.name)
            draft = self.latest(path.name)
            result.append(
                {
                    "session_id": path.name,
                    "start_utc": raw["start_utc"],
                    "duration_s": raw["duration_s"],
                    "synthetic": raw["synthetic"],
                    "mode": draft["detection_mode"],
                    "review_status": draft["review_status"],
                    "capabilities": raw["capabilities"],
                }
            )
        return sorted(result, key=lambda r: r["start_utc"], reverse=True)

    def edit(self, key: str, parent: str, operation: dict) -> dict:
        latest = self.latest(key)
        if latest["revision_id"] != parent:
            raise DataError("Revisione cambiata: ricarica prima di modificare; nessun edit perso")
        changed = apply_edit(latest, operation, self.context())
        return self._save(key, changed, latest["revision_id"])

    def suggestions(
        self, key: str, slot_id: str, muscle: str | None = None, *, evaluation: bool = False
    ) -> dict:
        raw = self.raw(key)
        draft = self.latest(key)
        slot = next((s for s in draft["slots"] if s["slot_id"] == slot_id), None)
        if slot is None:
            raise DataError("Slot sconosciuto")
        context = self.context()
        ids = {e["id"] for e in context["exercises"]}
        history_refs = []
        # Only confirmed slots from the latest prior revision; partial sessions are allowed.
        for session in self.sessions():
            if (
                session["session_id"] == key
                or session["start_utc"] >= raw["start_utc"]
                or session["synthetic"] != raw["synthetic"]
            ):
                continue
            old = self.latest(session["session_id"])
            previous = None
            for item in old["slots"]:
                if (
                    item["kind"] != "work"
                    or not item["confirmed"]
                    or item["exercise_id"] not in ids
                    or item.get("provenance") != "user_confirmed"
                    or item["start_s"] is None
                ):
                    continue
                stamp = datetime.fromisoformat(old["start_utc"]) + timedelta(
                    seconds=item["start_s"]
                )
                end = datetime.fromisoformat(old["start_utc"]) + timedelta(seconds=item["end_s"])
                if end >= datetime.fromisoformat(raw["start_utc"]):
                    continue
                context["history"].append(
                    {
                        "exercise_id": item["exercise_id"],
                        "timestamp_utc": stamp.isoformat(),
                        "reps": item["reps"],
                        "load_kg": item["external_load_kg"],
                        "duration_s": None
                        if item["start_s"] is None
                        else sum(b - a for a, b in item["work_intervals_s"]),
                        "previous_exercise_id": previous,
                    }
                )
                history_refs.append(
                    {
                        "session_id": session["session_id"],
                        "revision_id": old["revision_id"],
                        "slot_id": item["slot_id"],
                    }
                )
                previous = item["exercise_id"]
        preceding = [
            s
            for s in draft["slots"]
            if s["confirmed"]
            and s["kind"] == "work"
            and s["start_s"] is not None
            and slot["start_s"] is not None
            and s["end_s"] <= slot["start_s"]
        ]
        previous = max(preceding, key=lambda s: s["end_s"])["exercise_id"] if preceding else None
        quality = self.quality(key)
        excluded = set(quality["device_structure"]["invalid_set_ids"])
        excluded.update(i for check in quality["duration_checks"] for i in check["long_set_ids"])
        # Both ends of an overlap are ambiguous; retain their hints, suppress ranking use.
        sets = raw["device_sets"]
        valid = sorted(
            (
                r
                for r in sets
                if r["start_s"] is not None and r["end_s"] is not None and r["start_s"] < r["end_s"]
            ),
            key=lambda r: r["start_s"],
        )
        leader = None
        for row in valid:
            if leader is not None and row["start_s"] < leader["end_s"]:
                excluded.update((leader["device_set_id"], row["device_set_id"]))
            if leader is None or row["end_s"] > leader["end_s"]:
                leader = row
        if evaluation:
            previous = None
        native = []
        candidates = []
        for row in sets:
            if row["device_set_id"] in slot.get("device_set_refs", []):
                for candidate in normalize_categories(row["category_device"]):
                    usable = candidate["known_category"] and row["device_set_id"] not in excluded
                    candidates.append(
                        {
                            **candidate,
                            "device_set_id": row["device_set_id"],
                            "eligible_for_ranking": usable,
                        }
                    )
                    if usable:
                        native.append(candidate["native_name"])
        result = suggest_exercises(slot, context, raw["start_utc"], previous, muscle, native)
        result["native_candidates"] = candidates
        if evaluation:
            result["context_snapshot"] = context
            result["context_baseline"] = suggest_exercises(slot, context, raw["start_utc"])
            result["evaluation_basis"] = "target_session_excluded_from_history_and_adjacency"
        result["source_sha256"] = raw["source_sha256"]
        result["revision_id"] = draft["revision_id"]
        result["quality_id"] = quality["quality_id"]
        result["synthetic"] = raw["synthetic"]
        result["history_refs"] = history_refs
        result["context_sha256"] = hashlib.sha256(
            json.dumps(context, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        result["code_sha256"] = {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("reconciliation.py", "strength_catalog.py", "exercise_vocabulary.py")
        }
        if any(c["known_category"] and not c["eligible_for_ranking"] for c in candidates):
            result["reason_codes"].append("device_structure_caution")
        return result

    def export(self, key: str, kind: str = "json") -> bytes:
        draft = self.latest(key)
        if draft["review_status"] != "reviewed":
            raise DataError("Conferma/classifica i blocchi e finalizza prima di esportare")
        if kind == "json":
            return json.dumps(draft, ensure_ascii=False, allow_nan=False, indent=2).encode()
        if kind != "csv":
            raise DataError("Esportazione json oppure csv")
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        indices = {}
        for row in draft["slots"]:
            if row["kind"] != "work":
                continue
            if row["start_s"] is None or row["body_mass_kg"] is None:
                raise DataError(
                    "CSV v1 richiede tempi e massa corporea dichiarati per ogni serie; "
                    "JSON preserva gli ignoti"
                )
            exercise = row.get("exercise_snapshot")
            if exercise is None:
                raise DataError("Esercizio confermato non piÃ¹ nel catalogo: ripristina contesto")
            indices[row["exercise_id"]] = indices.get(row["exercise_id"], 0) + 1
            stamp = (
                datetime.fromisoformat(draft["start_utc"]) + timedelta(seconds=row["start_s"])
            ).isoformat()
            mode = exercise["load_mode"]
            writer.writerow(
                [
                    stamp,
                    "fit-" + key,
                    exercise["id"],
                    mode,
                    row["body_mass_kg"],
                    row["added_load_kg"] if mode == "bodyweight" else 0,
                    row["external_load_kg"] if mode == "external" else 0,
                    indices[row["exercise_id"]],
                    row["reps"],
                    row["rir"],
                    row["rpe"],
                ]
            )
        return stream.getvalue().encode()
