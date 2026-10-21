import copy
import json

import pytest

from hybrid_training_analytics.models import DataError
from hybrid_training_analytics.strength_sessions import StrengthSessions, declarations
from hybrid_training_analytics.strength_synthetic import strength_fit_bytes


def test_manual_empty_session_and_revisions(tmp_path):
    catalog = StrengthSessions(tmp_path)
    row = catalog.save(None, None, {}, synthetic=True)
    assert row["start_utc"] is None and row["source_ref"] is None
    assert row["selected_sets"] == [] and row["declarations"]["session_rpe"] is None
    assert row["declarations"]["body_scope"] is None and row["sensor"] is None
    assert not list(tmp_path.rglob("sets.csv"))
    before = copy.deepcopy(row)
    updated = catalog.save(row["session_id"], row["revision_id"], {"body_scope": "upper"})
    assert updated["synthetic"] and updated["parent_revision_id"] == row["revision_id"]
    assert row == before and len(list(catalog.root.rglob("*.json"))) == 2
    with pytest.raises(DataError, match="Revisione cambiata"):
        catalog.save(row["session_id"], row["revision_id"], {})
    assert len(list(catalog.root.rglob("*.json"))) == 2
    assert json.loads(catalog.export(row["session_id"])) == updated


def test_fit_no_sets_no_inherited_sensor_and_raw_immutable(tmp_path):
    catalog = StrengthSessions(tmp_path)
    key = catalog.fit.import_fit(strength_fit_bytes(), synthetic=True)
    raw_before = catalog.fit.raw(key)
    draft_before = catalog.fit.latest(key)
    row = catalog.view("fit-" + key)
    assert row["review_state"] == "unreviewed" and row["revision_id"] is None
    assert catalog.sessions() == []
    row = catalog.save(row["session_id"], None, {})
    assert row["review_state"] == "acknowledged" and row["elapsed_duration_s"] == 300
    assert row["hr_support_s"] == 300 and row["selected_sets"] == []
    assert row["sensor"] is None and row["declared_duration_s"] is None
    assert catalog.fit.raw(key) == raw_before and catalog.fit.latest(key) == draft_before
    assert len(catalog.sessions()) == 1


def test_declared_times_and_ux_opt_in_are_separate(tmp_path):
    catalog = StrengthSessions(tmp_path)
    row = catalog.save(
        None,
        None,
        {
            "start_utc": "2025-01-01T13:00:00+01:00",
            "end_utc": "2025-01-01T13:40:00+01:00",
            "session_rpe": 6,
            "sensor": "wrist",
            "measurement_opt_in": True,
            "annotation_elapsed_s": 23,
            "annotation_actions": 3,
        },
        synthetic=True,
    )
    assert row["start_utc"] == "2025-01-01T12:00:00+00:00"
    assert row["elapsed_duration_s"] is None and row["declared_duration_s"] == 2400
    metrics = catalog.ux_metrics()
    assert metrics["annotation_elapsed_mean_s"] == 23 and metrics["annotation_actions_mean"] == 3
    catalog.save(row["session_id"], row["revision_id"], {})
    assert catalog.ux_metrics()["annotation_elapsed_mean_s"] is None


@pytest.mark.parametrize(
    "values",
    [
        {"session_rpe": 0},
        {"session_rpe": float("nan")},
        {"sensor": "inherit"},
        {"start_utc": "2025-01-01T12:00:00"},
        {"start_utc": "2025-01-01T12:00:00+00:00"},
        {"start_utc": "2025-01-01T13:00:00+00:00", "end_utc": "2025-01-01T12:00:00+00:00"},
        {"selected_set_ids": ["a", "a"]},
        {"annotation_elapsed_s": 20},
        {"measurement_opt_in": True, "annotation_actions": 1.5},
        {"fake": 1},
    ],
)
def test_invalid_minimal_fields(values):
    with pytest.raises(DataError):
        declarations(values)


def test_selected_unconfirmed_rejected_known_exercise_does_not_fake_set(tmp_path):
    catalog = StrengthSessions(tmp_path)
    key = catalog.fit.import_fit(strength_fit_bytes(), synthetic=True)
    slot = next(s for s in catalog.fit.latest(key)["slots"] if s["kind"] == "work_candidate")
    with pytest.raises(DataError):
        catalog.save("fit-" + key, None, {"selected_set_ids": [slot["slot_id"]]})
    row = catalog.save("fit-" + key, None, {"remembered_exercise_ids": ["squat"]})
    assert row["remembered_exercises"][0]["id"] == "squat" and row["selected_sets"] == []
    assert not any(s["exercise_id"] for s in catalog.fit.latest(key)["slots"])


def test_confirmed_set_snapshot_and_time_override(tmp_path):
    catalog = StrengthSessions(tmp_path)
    key = catalog.fit.import_fit(strength_fit_bytes(), synthetic=True)
    work = next(s for s in catalog.fit.latest(key)["slots"] if s["kind"] == "work_candidate")

    def edit(op):
        return catalog.fit.edit(key, catalog.fit.latest(key)["revision_id"], op)

    edit({"type": "assign", "slot_id": work["slot_id"], "exercise_id": "squat"})
    draft = edit(
        {
            "type": "confirm",
            "slot_id": work["slot_id"],
            "values": {"reps": 8, "external_load_kg": 40},
        }
    )
    row = catalog.save(
        "fit-" + key,
        None,
        {
            "selected_set_ids": [work["slot_id"]],
            "start_utc": "2025-02-01T12:00:00+00:00",
            "end_utc": "2025-02-01T12:04:00+00:00",
        },
    )
    assert row["selected_sets"][0]["revision_id"] == draft["revision_id"]
    assert row["elapsed_duration_s"] == 300 and row["declared_duration_s"] == 240
    assert row["provenance"]["start_utc"] == "user_declared"
    assert row["detail_level"] == "selected_sets"
    edit({"type": "assign", "slot_id": work["slot_id"], "exercise_id": "curl"})
    assert (
        catalog.latest(row["session_id"])["selected_sets"][0]["snapshot"]["exercise_id"] == "squat"
    )


def test_unknown_session_and_path_rejected(tmp_path):
    catalog = StrengthSessions(tmp_path)
    for value in ("../uv.lock", "fit-" + "f" * 64):
        with pytest.raises((DataError, OSError)):
            catalog.view(value)


def csv_diary(tmp_path, session_id):
    from hybrid_training_analytics.local_catalog import LocalCatalog

    header = (
        "timestamp,session_id,exercise,load_mode,body_mass_kg,added_load_kg,"
        "external_load_kg,set_index,reps,rir,rpe\n"
    )
    row = f"2025-02-01T12:00:30+00:00,{session_id},squat,external,80,0,40,1,8,2,7\n"
    LocalCatalog(tmp_path).import_strength((header + row).encode())


def test_hybrid_zero_known_sets_is_unknown_and_context_record_clock(tmp_path):
    from hybrid_training_analytics.hybrid import hybrid_summary

    catalog = StrengthSessions(tmp_path)
    key = catalog.fit.import_fit(strength_fit_bytes(), synthetic=True)
    summary = catalog.save("fit-" + key, None, {"session_rpe": 6})
    run = {"run_id": "invented", "start_utc": "2025-02-02T12:00:00+00:00", "duration_s": 1800}
    payload = hybrid_summary([], {}, [run], strength_summaries=[summary])
    session = payload["strength_sessions"][0]
    assert session["known_external_volume_kg_reps"] is None and session["known_set_count"] == 0
    assert session["load"]["session_rpe_load_au"] == 30
    assert session["body_scope_declared"] is None
    week = next(iter(payload["weeks"].values()))
    assert week["strength_sessions"] == 1 and week["strength_sessions_without_known_sets"] == 1
    assert week["strength_observed_elapsed_duration_s"] == 300
    assert week["strength_observed_declared_duration_s"] is None
    preceding = payload["run_contexts"][0]["context"]["preceding_strength"][0]
    assert preceding["end_basis"] == "inferred_recorded_bounds"
    assert preceding["lower_body_exposure_declared"] is None


def test_hybrid_undated_manual_counted_outside_weeks(tmp_path):
    from hybrid_training_analytics.hybrid import hybrid_summary

    summary = StrengthSessions(tmp_path).save(None, None, {}, synthetic=True)
    payload = hybrid_summary([], {}, [], strength_summaries=[summary])
    assert payload["strength_session_count"] == 1 and payload["weeks"] == {}
    assert payload["undated_strength_session_ids"] == [summary["session_id"]]


def test_hybrid_csv_fit_identity_deduplicates_without_adding_selected_volume(tmp_path):
    from hybrid_training_analytics.local_catalog import LocalCatalog

    catalog = StrengthSessions(tmp_path)
    key = catalog.fit.import_fit(strength_fit_bytes(), synthetic=True)
    row = catalog.save("fit-" + key, None, {})
    csv_diary(tmp_path, row["session_id"])
    payload = LocalCatalog(tmp_path).hybrid()
    assert payload["strength_session_count"] == 1
    session = payload["strength_sessions"][0]
    assert session["merge_basis"] == "fit_csv_identity" and session["known_set_count"] == 1
    assert session["known_external_volume_kg_reps"] == 320
    assert session["whole_session_volume_kg_reps"] is None
    assert (
        next(iter(payload["weeks"].values()))["force_by_exercise"]["squat / external"][
            "external_volume_kg_reps"
        ]
        == 320
    )


def test_explicit_csv_link_and_duplicate_link_rejected(tmp_path):
    from hybrid_training_analytics.local_catalog import LocalCatalog

    catalog = StrengthSessions(tmp_path)
    csv_diary(tmp_path, "manual-csv")
    row = catalog.save(None, None, {"linked_csv_session_id": "manual-csv"}, synthetic=True)
    assert LocalCatalog(tmp_path).hybrid()["strength_session_count"] == 1
    assert (
        LocalCatalog(tmp_path).hybrid()["strength_sessions"][0]["merge_basis"]
        == "explicit_csv_link"
    )
    with pytest.raises(DataError, match="già collegata"):
        catalog.save(None, None, {"linked_csv_session_id": "manual-csv"}, synthetic=True)
    assert len(catalog.sessions()) == 1 and catalog.latest(row["session_id"]) == row


def test_nearby_csv_and_minimal_are_not_merged_by_time(tmp_path):
    from hybrid_training_analytics.local_catalog import LocalCatalog

    catalog = StrengthSessions(tmp_path)
    csv_diary(tmp_path, "unlinked")
    catalog.save(
        None,
        None,
        {"start_utc": "2025-02-01T12:00:00+00:00", "end_utc": "2025-02-01T12:05:00+00:00"},
        synthetic=True,
    )
    assert LocalCatalog(tmp_path).hybrid()["strength_session_count"] == 2


def test_hybrid_declared_duration_and_rpe_not_set_rpe(tmp_path):
    from hybrid_training_analytics.local_catalog import LocalCatalog

    catalog = StrengthSessions(tmp_path)
    csv_diary(tmp_path, "csv-known")
    row = catalog.save(
        None,
        None,
        {
            "linked_csv_session_id": "csv-known",
            "start_utc": "2025-02-01T12:00:00+00:00",
            "end_utc": "2025-02-01T12:20:00+00:00",
        },
        synthetic=True,
    )
    payload = LocalCatalog(tmp_path).hybrid()
    assert payload["strength_sessions"][0]["load"]["session_rpe_load_au"] is None
    row = catalog.save(
        row["session_id"], row["revision_id"], {**row["declarations"], "session_rpe": 5}
    )
    assert (
        LocalCatalog(tmp_path).hybrid()["strength_sessions"][0]["load"]["session_rpe_load_au"]
        == 100
    )


def test_duplicate_summary_payload_refused(tmp_path):
    from hybrid_training_analytics.hybrid import hybrid_summary

    row = StrengthSessions(tmp_path).save(None, None, {}, synthetic=True)
    with pytest.raises(DataError, match="duplicata"):
        hybrid_summary([], {}, [], strength_summaries=[row, row])


@pytest.mark.parametrize(
    "change",
    [
        {"elapsed_duration_s": -1},
        {"hr_support_s": 12},
        {"source_ref": "f" * 64},
        {"synthetic": "true"},
        {"start_utc": "2025-01-01T10:00:00"},
        {"declared_duration_s": 123},
    ],
)
def test_bad_export_summary_fails_before_aggregation(tmp_path, change):
    from hybrid_training_analytics.hybrid import hybrid_summary

    row = StrengthSessions(tmp_path).save(None, None, {}, synthetic=True)
    row.update(change)
    with pytest.raises(DataError):
        hybrid_summary([], {}, [], strength_summaries=[row])


def test_unconfirmed_exported_snapshot_never_contributes_volume(tmp_path):
    from hybrid_training_analytics.hybrid import hybrid_summary

    sessions = StrengthSessions(tmp_path)
    key = sessions.fit.import_fit(strength_fit_bytes(), synthetic=True)
    work = next(s for s in sessions.fit.latest(key)["slots"] if s["kind"] == "work_candidate")
    for operation in (
        {"type": "assign", "slot_id": work["slot_id"], "exercise_id": "bench_press"},
        {
            "type": "confirm",
            "slot_id": work["slot_id"],
            "values": {"reps": 8, "external_load_kg": 40},
        },
    ):
        sessions.fit.edit(key, sessions.fit.latest(key)["revision_id"], operation)
    row = sessions.save("fit-" + key, None, {"selected_set_ids": [work["slot_id"]]})
    row["selected_sets"][0]["snapshot"]["confirmed"] = False
    with pytest.raises(DataError):
        hybrid_summary([], {}, [], strength_summaries=[row])
