"""Auditable motion clustering, uncalibrated exercise ranking and non-destructive edits."""

from __future__ import annotations

import copy
import math
import re
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import pairwise
from statistics import median
from uuid import uuid4

from .exercise_vocabulary import VOCABULARY
from .models import DataError, number, utc_timestamp

MUSCLES = ("chest", "back", "shoulders", "arms", "core", "legs", "glutes")
DEFAULT_EXERCISES = [
    {
        "id": "bench_press",
        "name": "Panca piana",
        "muscles": ["chest", "arms"],
        "load_mode": "external",
        "aliases": ["bench_press"],
    },
    {
        "id": "lat_pulldown",
        "name": "Lat machine",
        "muscles": ["back", "arms"],
        "load_mode": "external",
        "aliases": ["pulldown"],
    },
    {
        "id": "squat",
        "name": "Squat",
        "muscles": ["legs", "glutes"],
        "load_mode": "external",
        "aliases": ["squat"],
    },
    {
        "id": "leg_press",
        "name": "Leg press",
        "muscles": ["legs", "glutes"],
        "load_mode": "external",
        "aliases": ["leg_press"],
    },
    {
        "id": "shoulder_press",
        "name": "Shoulder press",
        "muscles": ["shoulders", "arms"],
        "load_mode": "external",
        "aliases": ["shoulder_press"],
    },
    {
        "id": "curl",
        "name": "Curl",
        "muscles": ["arms"],
        "load_mode": "external",
        "aliases": ["curl"],
    },
    {
        "id": "plank",
        "name": "Plank",
        "muscles": ["core"],
        "load_mode": "bodyweight",
        "aliases": ["plank"],
    },
]


def finite(value, minimum=0, maximum=float("inf")) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or number(value, minimum=minimum, maximum=maximum) is None
    ):
        raise DataError("Numero non valido, non finito o fuori limite tecnico")
    return float(value)


@dataclass(frozen=True)
class DetectionConfig:
    window_s: float = 5
    max_accel_gap_s: float = 2
    min_accel_coverage: float = 0.6
    min_cluster_separation: float = 0.5
    max_hr_gap_s: float = 10

    def validate(self):
        finite(self.window_s, 1, 60)
        finite(self.max_accel_gap_s, 0.001, 60)
        finite(self.min_accel_coverage, 0.01, 1)
        finite(self.min_cluster_separation, 0.01, 1)
        finite(self.max_hr_gap_s, 0.001, 60)


def _cluster(values: list[float], separation: float) -> tuple[float, float] | None:
    if len(values) < 6:
        return None
    low, high = min(values), max(values)
    for _ in range(40):
        middle = (low + high) / 2
        left = [v for v in values if v <= middle]
        right = [v for v in values if v > middle]
        if not left or not right:
            return None
        a, b = sum(left) / len(left), sum(right) / len(right)
        if abs(a - low) + abs(b - high) < 1e-12:
            break
        low, high = a, b
    if high <= 0 or (high - low) / high < separation:
        return None
    return low, high


def _hr_mean(raw: dict, start: float, end: float, max_gap_s: float) -> tuple[float | None, float]:
    total = valid = 0.0
    for a, b in pairwise(raw["records"]):
        left, right = max(start, a["elapsed_s"]), min(end, b["elapsed_s"])
        dt = b["elapsed_s"] - a["elapsed_s"]
        if (
            left >= right
            or dt > max_gap_s
            or a["heart_rate_bpm"] is None
            or b["heart_rate_bpm"] is None
        ):
            continue
        av = (
            a["heart_rate_bpm"]
            + (b["heart_rate_bpm"] - a["heart_rate_bpm"]) * (left - a["elapsed_s"]) / dt
        )
        bv = (
            a["heart_rate_bpm"]
            + (b["heart_rate_bpm"] - a["heart_rate_bpm"]) * (right - a["elapsed_s"]) / dt
        )
        total += (av + bv) / 2 * (right - left)
        valid += right - left
    return (total / valid if valid else None, valid)


def _slot(start, end, kind, evidence, index):
    return {
        "slot_id": f"slot-{index}",
        "start_s": start,
        "end_s": end,
        "kind": kind,
        "work_intervals_s": [[start, end]] if kind in {"work_candidate", "work"} else [],
        "evidence": evidence,
        "source_slot_ids": [f"slot-{index}"],
        "exercise_id": None,
        "exercise_snapshot": None,
        "reps": None,
        "external_load_kg": None,
        "body_mass_kg": None,
        "added_load_kg": None,
        "rir": None,
        "rpe": None,
        "confirmed": False,
        "provenance": "algorithm_hypothesis",
        "device_reps": None,
        "device_weight_kg": None,
    }


def detect_slots(raw: dict, config: DetectionConfig | None = None) -> dict:
    config = config or DetectionConfig()
    config.validate()
    duration = finite(raw["duration_s"], 0.001, 86400)
    # Prefer one calibrated unit across the whole session, never mix g with counts.
    axes = ("accel_x_g", "accel_y_g", "accel_z_g")

    def complete(row, keys):
        return row["elapsed_s"] is not None and all(row[k] is not None for k in keys)

    if not any(complete(r, axes) for r in raw["accelerometer"]):
        axes = ("accel_x_counts", "accel_y_counts", "accel_z_counts")
    indexed = {
        r["elapsed_s"]: r
        for r in raw["accelerometer"]
        if complete(r, axes) and 0 <= r["elapsed_s"] <= duration
    }
    acceleration = sorted(indexed.values(), key=lambda r: r["elapsed_s"])
    bins = []
    start = 0.0
    # Single pass through sorted samples; no fabricated points or resampling.
    cursor = 0
    hr_records = [r for r in raw["records"] if 0 <= r["elapsed_s"] <= duration]
    hr_times = [r["elapsed_s"] for r in hr_records]
    while start < duration:
        end = min(start + config.window_s, duration)
        rows = []
        while cursor < len(acceleration) and acceleration[cursor]["elapsed_s"] < end:
            if acceleration[cursor]["elapsed_s"] >= start:
                rows.append(acceleration[cursor])
            cursor += 1
        coverage = (
            0 if len(rows) < 2 else (rows[-1]["elapsed_s"] - rows[0]["elapsed_s"]) / (end - start)
        )
        gap = max(
            (b["elapsed_s"] - a["elapsed_s"] for a, b in pairwise(rows)), default=float("inf")
        )
        energy = None
        if (
            len(rows) >= 3
            and coverage >= config.min_accel_coverage
            and gap <= config.max_accel_gap_s
        ):
            means = [sum(r[k] for r in rows) / len(rows) for k in axes]
            energy = math.sqrt(
                sum(sum((r[k] - m) ** 2 for k, m in zip(axes, means, strict=True)) for r in rows)
                / len(rows)
            )
        hr_rows = hr_records[
            max(0, bisect_left(hr_times, start) - 1) : bisect_right(hr_times, end) + 1
        ]
        hr, valid = _hr_mean({"records": hr_rows}, start, end, config.max_hr_gap_s)
        bins.append(
            {
                "start_s": start,
                "end_s": end,
                "motion_rms": energy,
                "motion_units": "g" if axes[0].endswith("_g") else "counts",
                "accel_coverage": coverage,
                "hr_mean_bpm": hr,
                "hr_valid_s": valid,
                "state": "unknown",
            }
        )
        start = end
    centres = _cluster(
        [b["motion_rms"] for b in bins if b["motion_rms"] is not None],
        config.min_cluster_separation,
    )
    mode = "motion_clusters" if centres else "device_hints" if raw["device_sets"] else "unresolved"
    if centres:
        threshold = sum(centres) / 2
        for row in bins:
            if row["motion_rms"] is not None:
                row["state"] = (
                    "work_candidate" if row["motion_rms"] > threshold else "rest_candidate"
                )
        # Remove isolated one-window flickers only between equal, known neighbors.
        old = [b["state"] for b in bins]
        for i in range(1, len(bins) - 1):
            if old[i - 1] == old[i + 1] and old[i - 1] != "unknown" and old[i] != "unknown":
                bins[i]["state"] = old[i - 1]
        slots = []
        for row in bins:
            if slots and slots[-1]["kind"] == row["state"]:
                slots[-1]["end_s"] = row["end_s"]
                if row["state"] == "work_candidate":
                    slots[-1]["work_intervals_s"] = [[slots[-1]["start_s"], row["end_s"]]]
            else:
                slots.append(
                    _slot(
                        row["start_s"],
                        row["end_s"],
                        row["state"],
                        ["motion_cluster_unvalidated"],
                        len(slots),
                    )
                )
    else:
        boundaries = {0.0, duration}
        usable = []
        for row in raw["device_sets"]:
            if (
                row["start_s"] is not None
                and row["end_s"] is not None
                and 0 <= row["start_s"] < row["end_s"] <= duration
            ):
                usable.append(row)
                boundaries.update((row["start_s"], row["end_s"]))
        slots = []
        for start, end in pairwise(sorted(boundaries)):
            covering = [s for s in usable if s["start_s"] <= start and s["end_s"] >= end]
            types = {s["device_set_type"] for s in covering}
            kind = (
                "work_candidate"
                if types == {"active"}
                else "rest_candidate"
                if types == {"rest"}
                else "unknown"
            )
            slots.append(
                _slot(
                    start,
                    end,
                    kind,
                    ["device_hint_not_verified"] if covering else ["no_motion_evidence"],
                    len(slots),
                )
            )
    for slot in slots:
        matches = []
        for row in raw["device_sets"]:
            if row["start_s"] is None or row["end_s"] is None:
                continue
            overlap = max(
                0, min(slot["end_s"], row["end_s"]) - max(slot["start_s"], row["start_s"])
            )
            if (
                overlap >= 0.8 * (slot["end_s"] - slot["start_s"])
                and row["device_set_type"] == "active"
            ):
                matches.append(row)
        slot["device_set_refs"] = [r["device_set_id"] for r in matches]
        if (
            len(matches) == 1
            and abs(matches[0]["duration_s"] - (slot["end_s"] - slot["start_s"])) <= config.window_s
        ):
            slot["device_reps"] = matches[0]["repetitions_device"]
            slot["device_weight_kg"] = matches[0]["weight_device_kg"]
    return {
        "schema_version": 1,
        "artifact_type": "strength_reconciliation",
        "algorithm": "motion_kmeans_v1",
        "source_sha256": raw["source_sha256"],
        "synthetic": raw["synthetic"],
        "start_utc": raw["start_utc"],
        "duration_s": duration,
        "detection_config": asdict(config),
        "detection_mode": mode,
        "cluster_centres": centres,
        "features": bins,
        "slots": slots,
        "edit_log": [],
        "review_status": "draft",
        "limits": [
            "Motion clusters are hypotheses, not measured work/rest or muscle activation",
            "HR is contextual only: no set boundaries or exercise inferred from HR",
            "No acceleration: missing laps unresolved; manual review required",
        ],
    }


def validate_context(context: dict) -> dict:
    if not isinstance(context, dict) or set(context) - {"exercises", "active_plan", "history"}:
        raise DataError("Contesto: campi sconosciuti")
    exercises = context.get("exercises", DEFAULT_EXERCISES)
    if not isinstance(exercises, list) or not 1 <= len(exercises) <= 300:
        raise DataError("Catalogo 1..300 esercizi")
    seen = set()
    for row in exercises:
        if not isinstance(row, dict) or set(row) != {
            "id",
            "name",
            "muscles",
            "load_mode",
            "aliases",
        }:
            raise DataError("Schema esercizio non valido")
        if any(not isinstance(row[k], str) or not 1 <= len(row[k]) <= 120 for k in ("id", "name")):
            raise DataError("Nome/ID esercizio non valido")
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,119}", row["id"]):
            raise DataError("ID esercizio: lettere, numeri, underscore, punto o trattino")
        if row["id"] in seen:
            raise DataError("ID esercizio duplicato")
        seen.add(row["id"])
        if row["load_mode"] not in {"external", "bodyweight"}:
            raise DataError("Modalità carico non valida")
        if (
            not isinstance(row["muscles"], list)
            or not row["muscles"]
            or any(m not in MUSCLES for m in row["muscles"])
        ):
            raise DataError("Gruppi muscolari non validi")
        if not isinstance(row["aliases"], list) or any(
            not isinstance(a, str) or len(a) > 120 for a in row["aliases"]
        ):
            raise DataError("Alias non validi")
    plan = context.get("active_plan", [])
    history = context.get("history", [])
    if not isinstance(plan, list) or any(e not in seen for e in plan):
        raise DataError("Scheda attiva: ID sconosciuti")
    if not isinstance(history, list) or len(history) > 10000:
        raise DataError("Storico non valido/oltre limite")
    result = []
    for row in history:
        if not isinstance(row, dict) or set(row) - {
            "exercise_id",
            "timestamp_utc",
            "reps",
            "load_kg",
            "duration_s",
            "previous_exercise_id",
            "confirmed",
        }:
            raise DataError("Schema storico non valido")
        if row.get("exercise_id") not in seen:
            raise DataError("Esercizio storico sconosciuto")
        if "confirmed" in row and not isinstance(row["confirmed"], bool):
            raise DataError("Conferma storico non valida")
        item = {**row, "timestamp_utc": utc_timestamp(row.get("timestamp_utc"))}
        for key in ("reps", "load_kg", "duration_s"):
            if item.get(key) is not None:
                finite(item[key])
        if (
            item.get("previous_exercise_id") is not None
            and item["previous_exercise_id"] not in seen
        ):
            raise DataError("Predecessore sconosciuto")
        result.append(item)
    return {"exercises": copy.deepcopy(exercises), "active_plan": list(plan), "history": result}


def suggest_exercises(
    slot: dict,
    context: dict,
    at_utc: str,
    previous: str | None = None,
    muscle: str | None = None,
    native_aliases: list[str] | None = None,
) -> dict:
    """Contextual alternatives with abstention, never an exercise recognition claim."""
    context = validate_context(context)
    if muscle is not None and muscle not in MUSCLES:
        raise DataError("Filtro muscolo non valido")
    at = datetime.fromisoformat(utc_timestamp(at_utc))
    aliases = set(native_aliases or [])
    reasons = []
    if "motion_cluster_unvalidated" not in slot.get("evidence", []):
        reasons.append("unavailable_motion")
    eligible = slot["kind"] in {"work", "work_candidate", "unknown"}
    ranked = []
    previous_groups = set(
        next((e["muscles"] for e in context["exercises"] if e["id"] == previous), [])
    )
    for exercise in context["exercises"]:
        if not eligible or (muscle and muscle not in exercise["muscles"]):
            continue
        history = []
        for entry in context["history"]:
            age = (at - datetime.fromisoformat(entry["timestamp_utc"])).total_seconds() / 86400
            if (
                entry["exercise_id"] == exercise["id"]
                and entry.get("confirmed", True)
                and 0 < age <= 90
            ):
                history.append((entry, math.exp(-age / 30)))
        parts = {
            "active_plan": 1.2 if exercise["id"] in context["active_plan"] else 0.0,
            "recent_history": math.log1p(sum(weight for _, weight in history)),
            "historical_transition": 0.5
            * math.log1p(
                sum(w for r, w in history if previous and r.get("previous_exercise_id") == previous)
            ),
            "adjacent_muscle": 0.2 if previous_groups.intersection(exercise["muscles"]) else 0.0,
            "native_alias": 0.3 if set(exercise["aliases"]).intersection(aliases) else 0.0,
        }
        # A filter or adjacency alone must not manufacture an exercise suggestion.
        if not any(parts[k] > 0 for k in ("active_plan", "recent_history", "native_alias")):
            continue
        declared = slot.get("confirmed") and slot.get("provenance") == "user_confirmed"
        for key, observed, history_key, weight in (
            (
                "duration_pattern",
                sum(b - a for a, b in slot["work_intervals_s"])
                if declared and slot["start_s"] is not None
                else None,
                "duration_s",
                0.3,
            ),
            ("reps_pattern", slot.get("reps") if declared else None, "reps", 0.4),
            (
                "load_pattern",
                slot.get("external_load_kg")
                if declared and exercise["load_mode"] == "external"
                else None,
                "load_kg",
                0.4,
            ),
        ):
            values = [r[history_key] for r, _ in history if r.get(history_key) is not None]
            parts[key] = 0.0
            if observed is not None and len(values) >= 3:
                centre = median(values)
                scale = max(
                    median(abs(v - centre) for v in values) * 1.4826, abs(centre) * 0.2, 1.0
                )
                parts[key] = weight * math.exp(-0.5 * ((observed - centre) / scale) ** 2)
        # No device volume feature: unverified kg/reps are not identifying evidence.
        parts["volume_pattern"] = 0.0
        explanation = []
        for key, text, source in (
            (
                "active_plan",
                "È tra gli esercizi della tua scheda; l’ordine è libero.",
                "user_declared",
            ),
            (
                "recent_history",
                f"Compare in {len(history)} serie precedenti confermate o dichiarate "
                "negli ultimi 90 giorni.",
                "user_declared",
            ),
            (
                "native_alias",
                "Una categoria Garmin è compatibile; resta un indizio non verificato.",
                "device_hint",
            ),
            (
                "historical_transition",
                "Nello storico dichiarato seguiva l’esercizio precedente.",
                "inferred",
            ),
            (
                "adjacent_muscle",
                "Condivide un gruppo del catalogo con l’esercizio precedente confermato.",
                "inferred",
            ),
            (
                "duration_pattern",
                "Durata dichiarata compatibile con almeno tre serie dello storico.",
                "inferred",
            ),
            (
                "reps_pattern",
                "Ripetizioni dichiarate compatibili con almeno tre serie dello storico.",
                "inferred",
            ),
            (
                "load_pattern",
                "Carico esterno dichiarato compatibile con almeno tre serie dello storico.",
                "inferred",
            ),
        ):
            if parts[key] > 0:
                explanation.append({"code": key, "text": text, "provenance": source})
        families = sorted(
            {f for alias in exercise["aliases"] for f in VOCABULARY.get(alias, ("", [], []))[2]}
        )
        ranked.append(
            {
                "exercise_id": exercise["id"],
                "name": exercise["name"],
                "muscles": exercise["muscles"],
                "movement_families": families,
                "score": sum(parts.values()),
                "components": parts,
                "explanations": explanation,
                "status": "possible" if parts["native_alias"] else "context_only",
            }
        )
    ranked.sort(key=lambda r: (-r["score"], r["name"], r["exercise_id"]))
    native_matches = [r for r in ranked if r["components"]["native_alias"]]
    # Preserve all device alternatives separately, including ones outside the top 3.
    conflict = len(native_matches) > 1
    if conflict:
        reasons.append("conflicting_candidates")
    if not ranked:
        reasons.append("not_work_block" if not eligible else "insufficient_evidence")
    elif not native_matches:
        reasons.append("context_only")
    denominator = sum(math.exp(r["score"] - ranked[0]["score"]) for r in ranked) if ranked else 0
    for row in ranked:
        # Legacy field, technical inspection only; never displayed as confidence.
        row["relative_weight"] = math.exp(row["score"] - ranked[0]["score"]) / denominator
    status = (
        "abstained"
        if not ranked
        else "conflicting_candidates"
        if conflict
        else "possible"
        if native_matches
        else "context_only"
    )
    text = {
        "abstained": "Non determinabile: mancano indizi utilizzabili per questo blocco.",
        "conflicting_candidates": (
            "Garmin propone alternative diverse: non possiamo scegliere quale sia vera."
        ),
        "possible": (
            "Possibili esercizi: indizi Garmin e contesto, da confermare solo se li ricordi."
        ),
        "context_only": (
            "Suggeriti dal contesto: scheda e storico "
            "non dimostrano cosa hai fatto in questo blocco."
        ),
    }[status]
    return {
        "artifact_type": "strength_exercise_suggestions",
        "schema_version": 1,
        "algorithm_version": "contextual_alternatives_v2",
        "top3": ranked[:3],
        "native_matches": [
            {
                "exercise_id": r["exercise_id"],
                "name": r["name"],
                "movement_families": r["movement_families"],
            }
            for r in native_matches
        ],
        "other_relative_weight": sum(r["relative_weight"] for r in ranked[3:]),
        "calibrated_probability": False,
        "automatic_assignment": False,
        "ranking_basis": (
            "contextual_alternatives_v2; plan + prior confirmed/declarative history; order free"
        ),
        "ranking_config": {
            "history_window_days": 90,
            "history_decay_days": 30,
            "plan_weight": 1.2,
            "native_weight": 0.3,
            "pattern_min_observations": 3,
            "transition_weight": 0.5,
            "adjacency_weight": 0.2,
            "duration_weight": 0.3,
            "reps_weight": 0.4,
            "load_weight": 0.4,
            "pattern_scale_mad_multiplier": 1.4826,
            "pattern_scale_relative_floor": 0.2,
            "pattern_scale_absolute_floor": 1.0,
        },
        "status": status,
        "reason_codes": reasons,
        "summary": text,
        "abstained": not ranked,
        "specific_evidence": bool(native_matches),
        "suppressed_features": []
        if slot.get("confirmed") and slot.get("provenance") == "user_confirmed"
        else ["unconfirmed_reps", "unconfirmed_load", "unconfirmed_duration"],
        "family_provenance": "inferred_product_vocabulary_not_measured_muscle_use",
    }


def validate_draft(draft: dict) -> None:
    duration = finite(draft["duration_s"], 0.001, 86400)
    ids = set()
    timed = []
    for slot in draft["slots"]:
        if slot["slot_id"] in ids:
            raise DataError("Slot ID duplicato")
        ids.add(slot["slot_id"])
        start, end = slot["start_s"], slot["end_s"]
        if (start is None) != (end is None):
            raise DataError("Tempi slot dichiarati insieme")
        if start is not None:
            finite(start)
            finite(end)
            if not 0 <= start < end <= duration:
                raise DataError("Slot fuori seduta o vuoto")
            timed.append((start, end))
            for a, b in slot["work_intervals_s"]:
                if not start <= a < b <= end:
                    raise DataError("Intervallo lavoro non valido")
        if slot["kind"] not in {
            "work_candidate",
            "rest_candidate",
            "work",
            "rest",
            "unknown",
            "excluded",
        }:
            raise DataError("Stato slot non valido")
        if not isinstance(slot["confirmed"], bool):
            raise DataError("Conferma non valida")
    timed.sort()
    if not timed or timed[0][0] != 0 or timed[-1][1] != duration:
        raise DataError("Timeline deve conservare l’intera durata")
    if any(b != c for (a, b), (c, d) in pairwise(timed)):
        raise DataError("Timeline con gap non rappresentati")
    if any(b > c for (a, b), (c, d) in pairwise(timed)):
        raise DataError("Slot temporali sovrapposti")


def apply_edit(draft: dict, operation: dict, context: dict) -> dict:
    if not isinstance(operation, dict):
        raise DataError("Operazione richiesta")
    result = copy.deepcopy(draft)
    slots = result["slots"]
    context = validate_context(context)
    kind = operation.get("type")
    by_id = {s["slot_id"]: s for s in slots}

    def get(key="slot_id"):
        if operation.get(key) not in by_id:
            raise DataError("Slot sconosciuto")
        return by_id[operation[key]]

    def reset(slot):
        slot.update(
            confirmed=False,
            reps=None,
            rir=None,
            rpe=None,
            device_reps=None,
            device_weight_kg=None,
            provenance="user_edit_pending_confirmation",
        )

    if kind == "add":
        slot = _slot(None, None, "work", ["manual_missing_set_no_invented_time"], uuid4().hex)
        slot["work_intervals_s"] = []
        slot["provenance"] = "user_added_unplaced"
        slots.append(slot)
    elif kind == "split":
        slot = get()
        cut = finite(operation.get("at_s"))
        if slot["start_s"] is None or not slot["start_s"] < cut < slot["end_s"]:
            raise DataError("Split deve essere interno al blocco")
        first, second = copy.deepcopy(slot), copy.deepcopy(slot)
        first["slot_id"] = uuid4().hex
        second["slot_id"] = uuid4().hex
        first["end_s"] = cut
        second["start_s"] = cut
        for item in (first, second):
            item["work_intervals_s"] = [
                [max(a, item["start_s"]), min(b, item["end_s"])]
                for a, b in slot["work_intervals_s"]
                if max(a, item["start_s"]) < min(b, item["end_s"])
            ]
            reset(item)
        index = slots.index(slot)
        slots[index : index + 1] = [first, second]
    elif kind == "merge":
        selected = operation.get("slot_ids")
        if (
            not isinstance(selected, list)
            or len(selected) < 2
            or len(set(selected)) != len(selected)
        ):
            raise DataError("Seleziona almeno due slot distinti")
        if any(key not in by_id for key in selected):
            raise DataError("Slot merge sconosciuto")
        if any(by_id[key]["start_s"] is None for key in selected):
            raise DataError("Merge richiede blocchi con tempi dichiarati")
        ordered = sorted([s for s in slots if s["start_s"] is not None], key=lambda s: s["start_s"])
        indices = sorted(ordered.index(by_id[key]) for key in selected)
        group = ordered[indices[0] : indices[-1] + 1]
        if any(a["end_s"] != b["start_s"] for a, b in pairwise(group)):
            raise DataError("Merge attraversa tempi non rappresentati")
        merged = _slot(
            group[0]["start_s"],
            group[-1]["end_s"],
            "work",
            ["manual_merge_rest_intervals_preserved"],
            uuid4().hex,
        )
        merged["work_intervals_s"] = [interval for s in group for interval in s["work_intervals_s"]]
        merged["source_slot_ids"] = sorted({key for s in group for key in s["source_slot_ids"]})
        merged["device_set_refs"] = sorted(
            {key for s in group for key in s.get("device_set_refs", [])}
        )
        slots[:] = [s for s in slots if s not in group] + [merged]
    elif kind == "place":
        slot = get()
        start = finite(operation.get("start_s"))
        end = finite(operation.get("end_s"))
        if slot["start_s"] is not None or not 0 <= start < end <= result["duration_s"]:
            raise DataError("Place richiede set senza tempi e intervallo valido")
        covering = [
            s
            for s in slots
            if s is not slot
            and s["start_s"] is not None
            and s["start_s"] <= start
            and s["end_s"] >= end
            and s["kind"] in {"rest_candidate", "rest", "unknown", "excluded"}
        ]
        if len(covering) != 1:
            raise DataError(
                "Posiziona dentro un unico blocco rest/unknown; nessuna sovrapposizione lavoro"
            )
        old = covering[0]
        slots.remove(old)
        for a, b in ((old["start_s"], start), (end, old["end_s"])):
            if a < b:
                slots.append(_slot(a, b, old["kind"], old["evidence"], uuid4().hex))
        slot.update(start_s=start, end_s=end, work_intervals_s=[[start, end]])
        reset(slot)
    elif kind == "assign":
        slot = get()
        exercise = operation.get("exercise_id")
        if exercise not in {e["id"] for e in context["exercises"]}:
            raise DataError("Esercizio sconosciuto")
        if slot["exercise_id"] != exercise:
            for key in ("reps", "external_load_kg", "added_load_kg", "rir", "rpe"):
                slot[key] = None
        slot["exercise_id"] = exercise
        slot["exercise_snapshot"] = copy.deepcopy(
            next(e for e in context["exercises"] if e["id"] == exercise)
        )
        slot["kind"] = "work"
        slot["confirmed"] = False
        if slot["start_s"] is not None and not slot["work_intervals_s"]:
            slot["work_intervals_s"] = [[slot["start_s"], slot["end_s"]]]
        slot["provenance"] = "user_assignment_pending_confirmation"
    elif kind == "confirm":
        slot = get()
        if slot["exercise_id"] is None:
            raise DataError("Seleziona prima l’esercizio")
        values = operation.get("values", {})
        if not isinstance(values, dict) or set(values) - {
            "reps",
            "external_load_kg",
            "body_mass_kg",
            "added_load_kg",
            "rir",
            "rpe",
        }:
            raise DataError("Campi dichiarazione serie non validi")
        exercise = slot["exercise_snapshot"]
        if exercise is None:
            raise DataError("Assegna nuovamente l’esercizio prima di confermare")
        for key in ("reps", "external_load_kg", "body_mass_kg", "added_load_kg", "rir", "rpe"):
            value = values.get(key)
            if value is not None:
                finite(
                    value,
                    1 if key in {"reps", "body_mass_kg", "rpe"} else 0,
                    10 if key in {"rir", "rpe"} else float("inf"),
                )
            slot[key] = value
        if slot["reps"] is None or int(slot["reps"]) != slot["reps"]:
            raise DataError("Ripetizioni intere dichiarate richieste")
        if exercise["load_mode"] == "external" and (
            slot["external_load_kg"] is None or slot["added_load_kg"] is not None
        ):
            raise DataError("Carico esterno dichiarato richiesto, zavorra separata")
        if exercise["load_mode"] == "bodyweight" and (
            slot["body_mass_kg"] is None
            or slot["added_load_kg"] is None
            or slot["external_load_kg"] is not None
        ):
            raise DataError("Massa corporea e zavorra dichiarate richieste, senza carico esterno")
        slot.update(confirmed=True, kind="work", provenance="user_confirmed")
    elif kind == "classify":
        slot = get()
        state = operation.get("kind")
        if state not in {"rest", "unknown", "excluded", "work"}:
            raise DataError("Classificazione non valida")
        slot.update(
            kind=state, confirmed=state in {"rest", "excluded"}, provenance="user_classification"
        )
        if state == "work" and slot["start_s"] is not None:
            slot["work_intervals_s"] = [[slot["start_s"], slot["end_s"]]]
        elif state != "work":
            slot["work_intervals_s"] = []
    elif kind == "finalize":
        if any(not s["confirmed"] for s in slots):
            raise DataError("Restano blocchi da confermare/classificare")
        result["review_status"] = "reviewed"
    else:
        raise DataError("Operazione non supportata")
    if kind != "finalize":
        result["review_status"] = "draft"
    result["slots"] = sorted(
        slots, key=lambda s: (s["start_s"] is None, s["start_s"] or 0, s["slot_id"])
    )
    result["edit_log"].append(copy.deepcopy(operation))
    validate_draft(result)
    return result
