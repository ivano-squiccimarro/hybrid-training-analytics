"""Opt-in frozen prediction trials; labels never become training history implicitly."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .annotations import next_sequence, revision_key
from .models import DataError
from .strength_catalog import StrengthCatalog
from .strength_sessions import StrengthSessions

FAMILIES = ("spinta", "tirata", "ginocchio", "anca", "core", "altro")


def trial_identity(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
        raise DataError("Identità della prova non valida")
    return value


def evaluation_metrics(trials: list[dict]) -> dict:
    count = len(trials)
    labelled = [r for r in trials if r.get("label", {}).get("exercise_id") is not None]
    family_labels = [r for r in trials if r.get("label", {}).get("movement_families")]
    attempts = sum(not r["suggestions"]["abstained"] for r in trials)

    def hit(row, basis):
        exercise = row["label"]["exercise_id"]
        return exercise in row[basis]

    ranked_hits = sum(hit(r, "predicted_ids") for r in labelled)
    attempted_labels = [r for r in labelled if not r["suggestions"]["abstained"]]
    family_hits = sum(
        bool(set(r["label"]["movement_families"]).intersection(r["predicted_families"]))
        for r in family_labels
    )
    corrections = [
        r["label"]["corrections"]
        for r in trials
        if r.get("label", {}).get("corrections") is not None
    ]
    usefulness = [
        r["label"]["usefulness"] for r in trials if r.get("label", {}).get("usefulness") is not None
    ]
    return {
        "artifact_type": "strength_suggestion_evaluation",
        "schema_version": 1,
        "synthetic": bool(trials) and all(r["synthetic"] for r in trials),
        "captured_blocks": count,
        "sessions": len({r["session_id"] for r in trials}),
        "labelled_exercise_blocks": len(labelled),
        "unknown_exercise_blocks": count - len(labelled),
        "suggested_blocks": attempts,
        "abstained_blocks": count - attempts,
        "suggestion_coverage_fraction": attempts / count if count else None,
        "abstention_fraction": (count - attempts) / count if count else None,
        "top3_hits": ranked_hits,
        "top3_misses_including_abstention": len(labelled) - ranked_hits,
        "top3_hit_fraction_all_labelled": ranked_hits / len(labelled) if labelled else None,
        "labelled_nonabstained_blocks": len(attempted_labels),
        "top3_hit_fraction_nonabstained_labelled": sum(
            hit(r, "predicted_ids") for r in attempted_labels
        )
        / len(attempted_labels)
        if attempted_labels
        else None,
        "garmin_baseline_hits": sum(hit(r, "garmin_ids") for r in labelled),
        "context_baseline_hits": sum(hit(r, "context_ids") for r in labelled),
        "labelled_family_blocks": len(family_labels),
        "top3_family_hits": family_hits,
        "top3_family_hit_fraction": family_hits / len(family_labels) if family_labels else None,
        "correction_observations": len(corrections),
        "declared_corrections_mean": sum(corrections) / len(corrections) if corrections else None,
        "usefulness_responses": len(usefulness),
        "usefulness_counts": {
            v: usefulness.count(v) for v in ("helpful", "not_helpful", "uncertain")
        },
        "provenance": "frozen_predictions_vs_independent_user_declared_labels",
        "limits": [
            "Synthetic tests do not validate real recognition",
            "Remembered labels are user declarations, not independently measured ground truth",
            "Family hit means any overlap of declared and suggested product families",
            "No success threshold, causal claim or generalization from one person",
            "Trials never become ranking history; target session excluded at capture",
        ],
    }


class StrengthEvaluation:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.root = self.workspace / "data" / "processed" / "hta-local" / "strength-evaluation"
        self.fit = StrengthCatalog(self.workspace)

    def frozen(self, trial_id: str) -> dict:
        path = self.root / trial_identity(trial_id) / "frozen.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        digest = row.pop("snapshot_sha256")
        if (
            hashlib.sha256(json.dumps(row, sort_keys=True, allow_nan=False).encode()).hexdigest()
            != digest
        ):
            raise DataError("Snapshot della prova modificato: conservato per diagnosi")
        return {**row, "snapshot_sha256": digest}

    def capture(self, session_id: str, slot_id: str) -> dict:
        for folder in self.root.glob("*"):
            if folder.is_dir():
                old = self.frozen(folder.name)
                if old["session_id"] == session_id and old["slot_id"] == slot_id:
                    return old
        draft = self.fit.latest(session_id)
        slot = next((s for s in draft["slots"] if s["slot_id"] == slot_id), None)
        if (
            slot is None
            or slot["confirmed"]
            or slot["exercise_id"] is not None
            or slot["kind"] not in {"work_candidate", "unknown", "work"}
        ):
            raise DataError(
                "Congela un blocco non ancora assegnato/confermato, prima di indicare la risposta"
            )
        suggestions = self.fit.suggestions(session_id, slot_id, evaluation=True)
        exercises = suggestions["context_snapshot"]["exercises"]
        native = []
        for candidate in suggestions["native_candidates"]:
            if candidate["known_category"]:
                for e in exercises:
                    if candidate["native_name"] in e["aliases"] and e["id"] not in native:
                        native.append(e["id"])
        row = {
            "artifact_type": "strength_suggestion_trial",
            "schema_version": 1,
            "trial_id": uuid4().hex,
            "session_id": session_id,
            "slot_id": slot_id,
            "captured_utc": datetime.now(UTC).isoformat(),
            "synthetic": draft["synthetic"],
            "slot_start_s": slot["start_s"],
            "slot_end_s": slot["end_s"],
            "suggestions": suggestions,
            "predicted_ids": [r["exercise_id"] for r in suggestions["top3"]],
            "predicted_families": sorted(
                {f for r in suggestions["top3"] for f in r["movement_families"]}
            ),
            "garmin_ids": native[:3],
            "context_ids": [r["exercise_id"] for r in suggestions["context_baseline"]["top3"]],
            "label_provenance": "not_yet_declared",
            "training_feedback": False,
        }
        row["snapshot_sha256"] = hashlib.sha256(
            json.dumps(row, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        folder = self.root / row["trial_id"]
        folder.mkdir(parents=True, exist_ok=False)
        with (folder / "frozen.json").open("x", encoding="utf-8") as out:
            json.dump(row, out, ensure_ascii=False, allow_nan=False, indent=2)
        return row

    def latest_label(self, trial_id: str) -> dict | None:
        self.frozen(trial_id)
        rows = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in (self.root / trial_id / "labels").glob("*.json")
        ]
        return max(rows, key=revision_key) if rows else None

    def label(self, trial_id: str, parent: str | None, values: dict) -> dict:
        frozen = self.frozen(trial_id)
        old = self.latest_label(trial_id)
        if (old["revision_id"] if old else None) != parent:
            raise DataError("Risposta cambiata: ricarica; nessuna etichetta sovrascritta")
        if not isinstance(values, dict) or set(values) - {
            "exercise_id",
            "movement_families",
            "usefulness",
            "corrections",
        }:
            raise DataError("Campi prova non validi")
        exercise = values.get("exercise_id")
        if exercise is not None and (
            not isinstance(exercise, str)
            or exercise
            not in {e["id"] for e in frozen["suggestions"]["context_snapshot"]["exercises"]}
        ):
            raise DataError("Esercizio fuori dal catalogo congelato")
        families = values.get("movement_families", [])
        if (
            not isinstance(families, list)
            or any(not isinstance(f, str) or f not in FAMILIES for f in families)
            or len(set(families)) != len(families)
        ):
            raise DataError("Famiglie dichiarate non valide")
        usefulness = values.get("usefulness")
        if usefulness is not None and (
            not isinstance(usefulness, str)
            or usefulness not in {"helpful", "not_helpful", "uncertain"}
        ):
            raise DataError("Riscontro utilità non valido")
        corrections = values.get("corrections")
        if corrections is not None and (
            type(corrections) is not int or not 0 <= corrections <= 1000
        ):
            raise DataError("Numero correzioni intero non negativo")
        row = {
            "trial_id": trial_id,
            "revision_id": uuid4().hex,
            "parent_revision_id": parent,
            "revision_sequence": next_sequence(self.root / trial_id / "labels"),
            "created_utc": datetime.now(UTC).isoformat(),
            "provenance": "user_declared_remembered_label",
            "values": {
                "exercise_id": exercise,
                "movement_families": families,
                "usefulness": usefulness,
                "corrections": corrections,
            },
        }
        folder = self.root / trial_id / "labels"
        folder.mkdir(exist_ok=True)
        with (folder / (row["revision_id"] + ".json")).open("x", encoding="utf-8") as out:
            json.dump(row, out, ensure_ascii=False, allow_nan=False, indent=2)
        return row

    def metrics(self) -> dict:
        trials = []
        for folder in sorted(self.root.glob("*")):
            if folder.is_dir():
                row = self.frozen(folder.name)
                label = self.latest_label(folder.name)
                row["label"] = label["values"] if label else {}
                trials.append(row)
        result = evaluation_metrics(trials)
        result["cohorts"] = {
            name: evaluation_metrics([r for r in trials if r["synthetic"] == flag])
            for name, flag in (("synthetic", True), ("local_real", False))
        }
        result["mixed_cohorts"] = bool(trials) and len({r["synthetic"] for r in trials}) > 1
        result["trial_refs"] = [
            {"trial_id": r["trial_id"], "snapshot_sha256": r["snapshot_sha256"]} for r in trials
        ]
        result["ux"] = StrengthSessions(self.workspace).ux_metrics()
        result["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return result
