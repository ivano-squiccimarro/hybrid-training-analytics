"""Concurrent training descriptors; proxies are never called physiological fatigue."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo

from .annotations import validate_annotations
from .models import DataError, check_timezone, number, utc_timestamp
from .strength import StrengthSet, summarize_strength


def session_load(values: dict, *, duration_s: float | None = None) -> dict:
    if duration_s is not None and number(duration_s, minimum=0.001) is None:
        raise DataError("Durata di sessione non valida")
    declared = validate_annotations(values)
    if declared["start_utc"] is not None:
        duration_s = (
            datetime.fromisoformat(declared["end_utc"])
            - datetime.fromisoformat(declared["start_utc"])
        ).total_seconds()
        basis = "user_declared_elapsed_session"
    else:
        basis = "record_elapsed_duration" if duration_s is not None else None
    rpe = declared["session_rpe"]
    return {
        "duration_s": duration_s,
        "duration_basis": basis,
        "session_rpe": rpe,
        "session_rpe_load_au": None if rpe is None or duration_s is None else rpe * duration_s / 60,
        "formula": "declared session RPE (1..10) * duration_s / 60",
        "provenance": "user_declaration_times_explicit_duration",
        "physiological_fatigue": None,
    }


def strength_trends(sets: list[StrengthSet], annotations: dict[str, dict]) -> list[dict]:
    groups = defaultdict(list)
    unknown = []
    for row in sets:
        declared = validate_annotations(annotations.get(row.session_id, {}))
        if declared["technique"] is None or row.rir is None:
            unknown.append(
                {
                    "session_id": row.session_id,
                    "exercise": row.exercise,
                    "set_index": row.set_index,
                    "status": "unmatched_missing_technique_or_rir",
                }
            )
            continue
        # Exact reps/RIR and explicit technique; no estimated 1RM or arbitrary aggregation.
        groups[(row.exercise, row.load_mode, declared["technique"], row.reps, row.rir)].append(row)
    result = []
    for (exercise, mode, technique, reps, rir), rows in groups.items():
        by_session = defaultdict(list)
        for row in rows:
            by_session[row.session_id].append(row)
        observed = []
        for session, session_rows in by_session.items():
            best = max(session_rows, key=lambda r: r.added_load_kg + r.external_load_kg)
            observed.append(
                {
                    "session_id": session,
                    "timestamp_utc": best.timestamp_utc,
                    "load_kg": best.added_load_kg + best.external_load_kg,
                    "body_mass_kg": best.body_mass_kg,
                    "sets_at_condition": len(session_rows),
                }
            )
        observed.sort(key=lambda r: (r["timestamp_utc"], r["session_id"]))
        for previous, current in zip(observed, observed[1:], strict=False):
            current["delta_load_kg"] = current["load_kg"] - previous["load_kg"]
            current["body_mass_changed"] = current["body_mass_kg"] != previous["body_mass_kg"]
        result.append(
            {
                "exercise": exercise,
                "load_mode": mode,
                "technique": technique,
                "reps": reps,
                "rir_declared": rir,
                "observations": observed,
                "interpretation": (
                    "Observed loads at declared matching conditions, not maximal strength"
                ),
            }
        )
    return result + unknown


def concurrent_context(
    run_start_utc: str, strength_sessions: list[dict], lookback_h: float = 72
) -> dict:
    if number(lookback_h, minimum=0.001) is None:
        raise DataError("Lookback non valido")
    run_start = datetime.fromisoformat(utc_timestamp(run_start_utc))
    preceding = []
    for session in strength_sessions:
        values = session["annotations"]
        end_utc = session.get("temporal_end_utc", values["end_utc"])
        if end_utc is None:
            continue
        end = datetime.fromisoformat(end_utc)
        hours = (run_start - end).total_seconds() / 3600
        if 0 <= hours <= lookback_h:
            preceding.append(
                {
                    "session_id": session["session_id"],
                    "hours_since_end": hours,
                    "lower_body_exposure_declared": values["lower_body_exposure"],
                    "body_scope_declared": session.get("body_scope_declared"),
                    "end_basis": session.get("end_basis", "user_declared"),
                    "elapsed_duration_s": session.get("elapsed_duration_s"),
                    "declared_duration_s": session.get("declared_duration_s"),
                    "known_sets": session.get("known_set_count", len(session.get("sets", []))),
                    "session_rpe_load_au": session["load"]["session_rpe_load_au"],
                }
            )
    preceding.sort(key=lambda row: row["hours_since_end"])
    return {
        "lookback_h": lookback_h,
        "preceding_strength": preceding,
        "strength_sessions_without_declared_end": sum(
            not s["annotations"]["end_utc"] for s in strength_sessions
        ),
        "strength_sessions_without_any_end": sum(
            s.get("temporal_end_utc", s["annotations"]["end_utc"]) is None
            for s in strength_sessions
        ),
        "recovery_score": None,
        "interference_score": None,
        "interpretation": (
            "Temporal context only; unknown end/exposure remains unknown; no "
            "causal or recovery verdict"
        ),
    }


def hybrid_summary(
    sets: list[StrengthSet],
    annotations: dict[str, dict],
    runs: list[dict],
    timezone: str = "Europe/Rome",
    *,
    strength_summaries: list[dict] | None = None,
) -> dict:
    check_timezone(timezone)
    sessions = defaultdict(list)
    for row in sets:
        sessions[row.session_id].append(row)
    strength_sessions = []
    for identity, rows in sessions.items():
        values = validate_annotations(annotations.get(identity, {}))
        if values["start_utc"] and any(
            not values["start_utc"] <= r.timestamp_utc <= values["end_utc"] for r in rows
        ):
            raise DataError("Serie forza fuori intervallo dichiarato")
        strength_sessions.append(
            {
                "session_id": identity,
                "annotations": values,
                "load": session_load(values),
                "exercises": summarize_strength(rows),
                "sets": [asdict(r) for r in rows],
            }
        )
    strength_sessions = merge_minimal_sessions(strength_sessions, strength_summaries or [])
    undated = []
    weekly = defaultdict(
        lambda: {
            "run_elapsed_duration_s": 0.0,
            "observed_record_distance_m": 0.0,
            "runs_missing_record_distance": 0,
            "observed_declared_target_distance_m": 0.0,
            "runs_missing_declared_target_distance": 0,
            "runs": 0,
            "strength_sessions": 0,
            "observed_rpe_load_au": 0.0,
            "sessions_missing_rpe_load": 0,
            "force_by_exercise": {},
            "strength_sessions_without_known_sets": 0,
            "strength_observed_elapsed_duration_s": 0.0,
            "strength_sessions_missing_elapsed_duration": 0,
            "strength_observed_declared_duration_s": 0.0,
            "strength_sessions_missing_declared_duration": 0,
            "strength_observed_hr_support_s": 0.0,
            "strength_sessions_missing_hr_support": 0,
        }
    )

    def week(stamp: str) -> str:
        date = datetime.fromisoformat(stamp).astimezone(ZoneInfo(timezone)).date()
        year, index, _ = date.isocalendar()
        return f"{year}-W{index:02d}"

    for session in strength_sessions:
        stamp = session["temporal_start_utc"]
        if stamp is None:
            undated.append(session["session_id"])
            continue
        bucket = weekly[week(stamp)]
        bucket["strength_sessions"] += 1
        bucket["strength_sessions_without_known_sets"] += not session["exercises"]
        for source, target, missing in (
            (
                "elapsed_duration_s",
                "strength_observed_elapsed_duration_s",
                "strength_sessions_missing_elapsed_duration",
            ),
            (
                "declared_duration_s",
                "strength_observed_declared_duration_s",
                "strength_sessions_missing_declared_duration",
            ),
            (
                "hr_support_s",
                "strength_observed_hr_support_s",
                "strength_sessions_missing_hr_support",
            ),
        ):
            value = session[source]
            bucket[target] += value if value is not None else 0
            bucket[missing] += value is None
        load = session["load"]["session_rpe_load_au"]
        bucket["observed_rpe_load_au"] += load if load is not None else 0
        bucket["sessions_missing_rpe_load"] += load is None
        for row in session["exercises"]:
            key = row["exercise"] + " / " + row["load_mode"]
            item = bucket["force_by_exercise"].setdefault(
                key,
                {
                    "sets": 0,
                    "reps": 0,
                    "external_volume_kg_reps": 0,
                    "system_mass_proxy_kg_reps": None,
                },
            )
            for field in ("sets", "reps", "external_volume_kg_reps"):
                item[field] += row[field]
            if row["system_mass_proxy_kg_reps"] is not None:
                item["system_mass_proxy_kg_reps"] = (item["system_mass_proxy_kg_reps"] or 0) + row[
                    "system_mass_proxy_kg_reps"
                ]
    run_contexts = []
    for run in runs:
        values = validate_annotations(run.get("annotations", {}))
        load = session_load(values, duration_s=run["duration_s"])
        bucket = weekly[week(run["start_utc"])]
        bucket["runs"] += 1
        bucket["run_elapsed_duration_s"] += run["duration_s"]
        distance = run.get("record_distance_delta_m")
        bucket["observed_record_distance_m"] += distance if distance is not None else 0
        bucket["runs_missing_record_distance"] += distance is None
        target = values["target_distance_m"]
        bucket["observed_declared_target_distance_m"] += target if target is not None else 0
        bucket["runs_missing_declared_target_distance"] += target is None
        au = load["session_rpe_load_au"]
        bucket["observed_rpe_load_au"] += au if au is not None else 0
        bucket["sessions_missing_rpe_load"] += au is None
        run_contexts.append(
            {
                "run_id": run["run_id"],
                "load": load,
                "cardiovascular_exposure": run.get("cardiovascular_exposure"),
                "context": concurrent_context(run["start_utc"], strength_sessions),
            }
        )
    for bucket in weekly.values():
        for target, missing in (
            ("strength_observed_elapsed_duration_s", "strength_sessions_missing_elapsed_duration"),
            (
                "strength_observed_declared_duration_s",
                "strength_sessions_missing_declared_duration",
            ),
            ("strength_observed_hr_support_s", "strength_sessions_missing_hr_support"),
        ):
            if bucket[missing] == bucket["strength_sessions"]:
                bucket[target] = None
        if bucket["runs_missing_record_distance"] == bucket["runs"]:
            bucket["observed_record_distance_m"] = None
        if bucket["runs_missing_declared_target_distance"] == bucket["runs"]:
            bucket["observed_declared_target_distance_m"] = None
        bucket["rpe_load_complete"] = bucket["sessions_missing_rpe_load"] == 0
        if bucket["sessions_missing_rpe_load"] == bucket["runs"] + bucket["strength_sessions"]:
            bucket["observed_rpe_load_au"] = None
    return {
        "schema_version": 1,
        "artifact_type": "hybrid_descriptive_load",
        "synthetic": bool(strength_sessions or runs)
        and all(r.get("synthetic") is True for r in [*strength_sessions, *runs]),
        "timezone_for_week": timezone,
        "strength_sessions": strength_sessions,
        "strength_session_count": len(strength_sessions),
        "undated_strength_session_ids": undated,
        "strength_summary_refs": [
            {
                "session_id": r["session_id"],
                "revision_id": r["revision_id"],
                "quality_ref": r["quality_ref"],
            }
            for r in strength_summaries or []
        ],
        "strength_trends": strength_trends(sets, annotations),
        "run_contexts": run_contexts,
        "weeks": dict(sorted(weekly.items())),
        "limits": [
            "RPE missing is not zero; partial observed sum labelled incomplete",
            "Force volume is a per-exercise proxy, not neuromuscular fatigue",
            "Heart-rate exposure is not TRIMP or a personalized zone load",
            "No causal interference or recovered/not-recovered score",
        ],
    }


def snapshot_exercises(summary: dict) -> list[dict]:
    """Volume of selected declared sets only; missing body mass stays missing."""
    groups = defaultdict(list)
    for ref in summary["selected_sets"]:
        slot = ref["snapshot"]
        groups[(slot["exercise_id"], slot["exercise_snapshot"]["load_mode"])].append(slot)
    result = []
    for (exercise, mode), rows in groups.items():
        external = [
            (r["external_load_kg"] if mode == "external" else r["added_load_kg"]) * r["reps"]
            for r in rows
        ]
        masses = (
            [
                (r["body_mass_kg"] + r["added_load_kg"]) * r["reps"]
                if r["body_mass_kg"] is not None
                else None
                for r in rows
            ]
            if mode == "bodyweight"
            else []
        )
        result.append(
            {
                "session_id": summary["session_id"],
                "exercise": exercise,
                "exercise_name": rows[0]["exercise_snapshot"]["name"],
                "load_mode": mode,
                "sets": len(rows),
                "reps": sum(r["reps"] for r in rows),
                "external_volume_kg_reps": sum(external),
                "system_mass_proxy_kg_reps": sum(masses)
                if masses and all(v is not None for v in masses)
                else None,
                "volume_scope": "selected_confirmed_sets_only_not_whole_session",
            }
        )
    return result


def merge_minimal_sessions(legacy: list[dict], summaries: list[dict]) -> list[dict]:
    """Explicit identity/link only, never merge by nearby timestamps."""
    from .strength_sessions import declarations, session_identity, validate_summary

    result = []
    csv = {r["session_id"]: r for r in legacy}
    claimed, seen = set(), set()
    for summary in summaries:
        validate_summary(summary)
        key = session_identity(summary["session_id"])
        if key in seen:
            raise DataError("Seduta minima duplicata: usare soltanto l’ultima revisione")
        seen.add(key)
        clean = declarations(summary["declarations"])
        link = clean["linked_csv_session_id"] or (key if key in csv else None)
        if link is not None and (link not in csv or link in claimed):
            raise DataError("Collegamento CSV assente o duplicato")
        old = csv.get(link)
        claimed.add(link)
        csv_values = old["annotations"] if old else {}
        values = dict(csv_values)
        for field in ("session_rpe", "start_utc", "end_utc", "sensor", "notes"):
            if clean[field] is not None:
                values[field] = clean[field]
        values = validate_annotations(values)
        load = session_load(values, duration_s=summary["elapsed_duration_s"])
        if (
            summary["provenance"]["elapsed_duration_s"] == "inferred"
            and values["start_utc"] is None
        ):
            load["duration_basis"] = "inferred_record_bounds"
        start = clean["start_utc"] or csv_values.get("start_utc") or summary["start_utc"]
        end = clean["end_utc"] or csv_values.get("end_utc") or summary["end_utc"]
        exercises = old["exercises"] if old else snapshot_exercises(summary)
        for row in exercises:
            row["volume_scope"] = (
                "imported_known_csv_sets_only" if old else "selected_confirmed_sets_only"
            )
        result.append(
            {
                "session_id": key,
                "csv_session_id": link,
                "summary_revision_id": summary["revision_id"],
                "source_ref": summary["source_ref"],
                "synthetic": summary["synthetic"],
                "annotations": values,
                "csv_annotations": csv_values if old else None,
                "summary_declarations": clean,
                "body_scope_declared": clean["body_scope"],
                "temporal_start_utc": start,
                "temporal_end_utc": end,
                "end_basis": "user_declared"
                if clean["end_utc"] or csv_values.get("end_utc")
                else summary["provenance"]["end_utc"],
                "elapsed_duration_s": summary["elapsed_duration_s"],
                "declared_duration_s": summary["declared_duration_s"]
                or (old["load"]["duration_s"] if old else None),
                "hr_support_s": summary["hr_support_s"],
                "load": load,
                "exercises": exercises,
                "sets": old["sets"] if old else [],
                "known_set_count": len(old["sets"]) if old else len(summary["selected_sets"]),
                "selected_set_refs": summary["selected_sets"],
                "known_external_volume_kg_reps": sum(
                    r["external_volume_kg_reps"] for r in exercises
                )
                if exercises
                else None,
                "whole_session_volume_kg_reps": None,
                "volume_scope": "known_sets_only_whole_session_unknown",
                "merge_basis": "explicit_csv_link"
                if clean["linked_csv_session_id"]
                else "fit_csv_identity"
                if old
                else "minimal_only",
                "limits": [
                    "Recorded clock end may include forgotten periods; not last true work set",
                    "Linked CSV takes precedence for set volume; selected sets not added twice",
                ],
            }
        )
    for key, old in csv.items():
        if key in claimed:
            continue
        values = old["annotations"]
        result.append(
            {
                **old,
                "csv_session_id": key,
                "body_scope_declared": None,
                "temporal_start_utc": values["start_utc"]
                or min(r["timestamp_utc"] for r in old["sets"]),
                "temporal_end_utc": values["end_utc"],
                "end_basis": "user_declared" if values["end_utc"] else "unavailable",
                "elapsed_duration_s": None,
                "declared_duration_s": old["load"]["duration_s"],
                "hr_support_s": None,
                "known_set_count": len(old["sets"]),
                "known_external_volume_kg_reps": sum(
                    r["external_volume_kg_reps"] for r in old["exercises"]
                ),
                "whole_session_volume_kg_reps": None,
                "volume_scope": "imported_known_csv_sets_only",
            }
        )
    return result
