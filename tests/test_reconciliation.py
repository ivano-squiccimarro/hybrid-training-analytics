import copy

import pytest

from hybrid_training_analytics.models import DataError
from hybrid_training_analytics.reconciliation import (
    apply_edit,
    detect_slots,
    suggest_exercises,
    validate_context,
)
from hybrid_training_analytics.strength_fit import read_strength_fit
from hybrid_training_analytics.strength_synthetic import strength_fit_bytes


def draft(tmp_path):
    path = tmp_path / "s.fit"
    path.write_bytes(strength_fit_bytes())
    return detect_slots(read_strength_fit(path, synthetic=True))


def test_split_merge_preserve_raw_and_internal_rest(tmp_path):
    original = draft(tmp_path)
    before = copy.deepcopy(original)
    work = [s for s in original["slots"] if s["kind"] == "work_candidate"]
    merged = apply_edit(original, {"type": "merge", "slot_ids": [s["slot_id"] for s in work]}, {})
    slot = next(s for s in merged["slots"] if s["kind"] == "work")
    assert (slot["start_s"], slot["end_s"]) == (30, 210)
    assert slot["work_intervals_s"] == [[30, 60], [180, 210]]
    assert sum(b - a for a, b in slot["work_intervals_s"]) == 60
    assert slot["reps"] is None
    split = apply_edit(merged, {"type": "split", "slot_id": slot["slot_id"], "at_s": 120}, {})
    assert sum(sum(b - a for a, b in s["work_intervals_s"]) for s in split["slots"]) == 60
    assert original == before
    assert split["review_status"] == "draft"


def test_missing_set_unplaced_until_explicit_times(tmp_path):
    original = draft(tmp_path)
    new = apply_edit(original, {"type": "add"}, {})
    slot = next(s for s in new["slots"] if s["start_s"] is None)
    assert slot["end_s"] is None and slot["reps"] is None
    placed = apply_edit(
        new, {"type": "place", "slot_id": slot["slot_id"], "start_s": 100, "end_s": 125}, {}
    )
    assert next(s for s in placed["slots"] if s["slot_id"] == slot["slot_id"])[
        "work_intervals_s"
    ] == [[100, 125]]
    with pytest.raises(DataError):
        apply_edit(
            new, {"type": "place", "slot_id": slot["slot_id"], "start_s": 45, "end_s": 55}, {}
        )


def test_rank_unordered_plan_explained_and_not_probability(tmp_path):
    original = draft(tmp_path)
    slot = original["slots"][1]
    context = validate_context({"active_plan": ["leg_press", "bench_press"]})
    context["history"] = [
        {
            "exercise_id": "leg_press",
            "timestamp_utc": "2025-01-31T12:00:00+00:00",
            "reps": 10,
            "load_kg": 50,
            "duration_s": 30,
        }
        for _ in range(3)
    ]
    scored = suggest_exercises(slot, context, original["start_utc"])
    assert scored["top3"][0]["exercise_id"] == "leg_press"
    assert scored["top3"][0]["components"]["reps_pattern"] == 0
    assert scored["calibrated_probability"] is False
    assert sum(r["relative_weight"] for r in scored["top3"]) + scored[
        "other_relative_weight"
    ] == pytest.approx(1)
    assert (
        suggest_exercises(slot, context, original["start_utc"], muscle="chest")["top3"][0][
            "exercise_id"
        ]
        == "bench_press"
    )
    future = {
        **context,
        "history": [{"exercise_id": "bench_press", "timestamp_utc": "2026-01-01T12:00:00+00:00"}]
        * 50,
    }
    assert (
        suggest_exercises(slot, future, original["start_utc"])["top3"][0]["components"][
            "recent_history"
        ]
        == 0
    )


def test_assign_confirm_and_finalize(tmp_path):
    current = draft(tmp_path)
    work = next(s for s in current["slots"] if s["kind"] == "work_candidate")
    with pytest.raises(DataError):
        apply_edit(current, {"type": "finalize"}, {})
    current = apply_edit(
        current, {"type": "assign", "slot_id": work["slot_id"], "exercise_id": "bench_press"}, {}
    )
    with pytest.raises(DataError):
        apply_edit(
            current, {"type": "confirm", "slot_id": work["slot_id"], "values": {"reps": 10}}, {}
        )
    current = apply_edit(
        current,
        {
            "type": "confirm",
            "slot_id": work["slot_id"],
            "values": {"reps": 10, "external_load_kg": 50},
        },
        {},
    )
    assert next(s for s in current["slots"] if s["slot_id"] == work["slot_id"])["rir"] is None
    for slot in current["slots"]:
        if not slot["confirmed"]:
            current = apply_edit(
                current, {"type": "classify", "slot_id": slot["slot_id"], "kind": "excluded"}, {}
            )
    final = apply_edit(current, {"type": "finalize"}, {})
    assert final["review_status"] == "reviewed"


@pytest.mark.parametrize(
    "edit",
    [
        {"type": "split", "slot_id": "missing", "at_s": 5},
        {"type": "merge", "slot_ids": ["slot-1", "slot-1"]},
        {"type": "classify", "slot_id": "slot-0", "kind": "invented"},
        {"type": "assign", "slot_id": "slot-1", "exercise_id": "unknown"},
    ],
)
def test_invalid_edits_do_not_mutate(tmp_path, edit):
    original = draft(tmp_path)
    before = copy.deepcopy(original)
    with pytest.raises(DataError):
        apply_edit(original, edit, {})
    assert original == before
