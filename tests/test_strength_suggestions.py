"""Invented inputs only: abstention, leakage, noisy hints and append-only history."""

import copy
from datetime import datetime, timedelta

import pytest

from hybrid_training_analytics.reconciliation import suggest_exercises, validate_context
from hybrid_training_analytics.strength_catalog import StrengthCatalog
from hybrid_training_analytics.strength_synthetic import strength_fit_bytes

AT = "2025-02-01T12:00:00+00:00"


def slot():
    return {
        "kind": "work_candidate",
        "start_s": 30,
        "end_s": 60,
        "work_intervals_s": [[30, 60]],
        "confirmed": False,
        "provenance": "algorithm_hypothesis",
        "evidence": [],
        "device_reps": 10,
        "device_weight_kg": 50,
    }


def test_no_evidence_or_filter_or_adjacency_does_not_invent_top3():
    for kwargs in ({}, {"muscle": "chest"}, {"previous": "bench_press"}):
        result = suggest_exercises(slot(), {}, AT, **kwargs)
        assert result["abstained"] and result["top3"] == []
        assert "insufficient_evidence" in result["reason_codes"]
        assert not result["specific_evidence"]


def test_plan_only_is_context_and_order_free():
    a = suggest_exercises(slot(), {"active_plan": ["squat", "bench_press"]}, AT)
    b = suggest_exercises(slot(), {"active_plan": ["bench_press", "squat"]}, AT)
    assert a == b
    assert a["status"] == "context_only" and not a["specific_evidence"]
    assert {r["exercise_id"] for r in a["top3"]} == {"squat", "bench_press"}
    assert all(r["explanations"][0]["provenance"] == "user_declared" for r in a["top3"])


def test_conflicting_native_alternatives_and_families_preserved():
    result = suggest_exercises(slot(), {}, AT, native_aliases=["bench_press", "squat"])
    assert result["status"] == "conflicting_candidates"
    assert {r["exercise_id"] for r in result["native_matches"]} == {"squat", "bench_press"}
    assert {f for r in result["native_matches"] for f in r["movement_families"]} == {
        "spinta",
        "ginocchio",
        "anca",
    }
    assert not result["automatic_assignment"] and not result["calibrated_probability"]


def test_filter_never_strengthens_single_match_confidence():
    result = suggest_exercises(slot(), {}, AT, muscle="chest", native_aliases=["bench_press"])
    assert len(result["top3"]) == 1
    assert result["top3"][0]["status"] == "possible"
    assert not result["calibrated_probability"]
    assert suggest_exercises(slot(), {}, AT, muscle="legs", native_aliases=["bench_press"])[
        "abstained"
    ]


@pytest.mark.parametrize("kind", ["rest", "rest_candidate", "excluded"])
def test_rest_and_excluded_have_no_exercise_suggestions(kind):
    result = suggest_exercises({**slot(), "kind": kind}, {"active_plan": ["squat"]}, AT)
    assert result["top3"] == [] and "not_work_block" in result["reason_codes"]


def test_future_same_instant_old_and_unconfirmed_history_ignored():
    history = [
        {"exercise_id": "squat", "timestamp_utc": t, "confirmed": flag}
        for t, flag in [
            ("2025-02-02T12:00:00+00:00", True),
            (AT, True),
            ("2024-01-01T12:00:00+00:00", True),
            ("2025-01-31T12:00:00+00:00", False),
        ]
    ]
    assert suggest_exercises(slot(), {"history": history}, AT)["abstained"]
    assert validate_context({"history": history})["history"] == history


def test_noisy_device_numbers_cannot_change_ranking_and_raw_not_mutated():
    context = {
        "history": [
            {
                "exercise_id": "squat",
                "timestamp_utc": "2025-01-31T12:00:00+00:00",
                "reps": 10,
                "load_kg": 50,
                "duration_s": 30,
            }
        ]
        * 3
    }
    original = slot()
    before = copy.deepcopy(original)
    a = suggest_exercises(original, context, AT)
    b = suggest_exercises(
        {**original, "device_reps": 999, "device_weight_kg": 9999, "work_intervals_s": [[30, 299]]},
        context,
        AT,
    )
    assert a == b and original == before
    assert a["top3"][0]["components"]["reps_pattern"] == 0
    declared = {
        **original,
        "confirmed": True,
        "provenance": "user_confirmed",
        "reps": 10,
        "external_load_kg": 50,
    }
    assert suggest_exercises(declared, context, AT)["top3"][0]["components"]["reps_pattern"] > 0


def test_quality_suppresses_ambiguous_device_hints_without_erasing(tmp_path):
    catalog = StrengthCatalog(tmp_path)
    key = catalog.import_fit(
        strength_fit_bytes(acceleration=False, category_codes=[0, 28], noisy=True), synthetic=True
    )
    draft = catalog.latest(key)
    raw = catalog.raw(key)
    # First two device intervals overlap: every associated category stays visible, unused.
    target = next(s for s in draft["slots"] if s["start_s"] == 0)
    result = catalog.suggestions(key, target["slot_id"])
    assert result["abstained"]
    assert result["native_candidates"] and all(
        not c["eligible_for_ranking"] for c in result["native_candidates"]
    )
    assert "device_structure_caution" in result["reason_codes"]
    assert catalog.latest(key) == draft and catalog.raw(key) == raw


def test_partial_confirmation_reused_latest_revision_only(tmp_path):
    catalog = StrengthCatalog(tmp_path)
    earlier = catalog.import_fit(strength_fit_bytes(), synthetic=True)
    # Distinct binary fixture; advance its recorded start for a prior-history scenario.
    later = catalog.import_fit(strength_fit_bytes(category_codes=[28]), synthetic=True)
    path = catalog.root / later / "raw.json"
    import json

    raw = json.loads(path.read_text())
    raw["start_utc"] = (datetime.fromisoformat(AT) + timedelta(days=1)).isoformat()
    path.write_text(json.dumps(raw))
    target = next(s for s in catalog.latest(later)["slots"] if s["kind"] == "work_candidate")
    work = next(s for s in catalog.latest(earlier)["slots"] if s["kind"] == "work_candidate")

    def edit(op):
        return catalog.edit(earlier, catalog.latest(earlier)["revision_id"], op)

    edit({"type": "assign", "slot_id": work["slot_id"], "exercise_id": "leg_press"})
    assert not any(
        r["exercise_id"] == "leg_press"
        for r in catalog.suggestions(later, target["slot_id"])["top3"]
    )
    edit(
        {
            "type": "confirm",
            "slot_id": work["slot_id"],
            "values": {"reps": 8, "external_load_kg": 60},
        }
    )
    assert catalog.latest(earlier)["review_status"] == "draft"
    result = catalog.suggestions(later, target["slot_id"])
    assert any(r["exercise_id"] == "leg_press" for r in result["top3"])
    edit({"type": "assign", "slot_id": work["slot_id"], "exercise_id": "curl"})
    assert not any(
        r["exercise_id"] == "leg_press"
        for r in catalog.suggestions(later, target["slot_id"])["top3"]
    )
    assert not any(
        r["exercise_id"] == "curl" for r in catalog.suggestions(later, target["slot_id"])["top3"]
    )


def test_prior_session_crossing_current_start_does_not_leak(tmp_path):
    catalog = StrengthCatalog(tmp_path)
    earlier = catalog.import_fit(strength_fit_bytes(), synthetic=True)
    later = catalog.import_fit(strength_fit_bytes(category_codes=[28]), synthetic=True)
    work = next(s for s in catalog.latest(earlier)["slots"] if s["kind"] == "work_candidate")
    row = catalog.latest(earlier)
    row["start_utc"] = (datetime.fromisoformat(AT) - timedelta(seconds=10)).isoformat()
    target = next(s for s in catalog.latest(later)["slots"] if s["kind"] == "work_candidate")
    oldslot = next(s for s in row["slots"] if s["slot_id"] == work["slot_id"])
    oldslot.update(
        exercise_id="leg_press",
        kind="work",
        confirmed=True,
        provenance="user_confirmed",
        reps=8,
        external_load_kg=60,
    )
    catalog._save(earlier, row, row["revision_id"])
    # Raw start must also precede target; invented catalog projection only.
    import json

    path = catalog.root / earlier / "raw.json"
    raw = catalog.raw(earlier)
    raw["start_utc"] = row["start_utc"]
    path.write_text(json.dumps(raw))
    assert not any(
        r["exercise_id"] == "leg_press"
        for r in catalog.suggestions(later, target["slot_id"])["top3"]
    )
