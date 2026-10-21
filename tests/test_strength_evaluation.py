"""Frozen trials on generated FIT only; null denominators and no feedback leakage."""

import copy
import json

import pytest

from hybrid_training_analytics.models import DataError
from hybrid_training_analytics.strength_catalog import StrengthCatalog
from hybrid_training_analytics.strength_evaluation import StrengthEvaluation, evaluation_metrics
from hybrid_training_analytics.strength_sessions import StrengthSessions
from hybrid_training_analytics.strength_synthetic import strength_fit_bytes


def fixture_trial(workspace):
    catalog = StrengthCatalog(workspace)
    key = catalog.import_fit(
        strength_fit_bytes(acceleration=False, device_sets=False), synthetic=True
    )
    context = catalog.context()
    context["active_plan"] = ["squat"]
    catalog.save_context(context)
    target = catalog.latest(key)["slots"][0]
    evaluation = StrengthEvaluation(workspace)
    return catalog, evaluation, key, target, evaluation.capture(key, target["slot_id"])


def test_frozen_first_capture_no_label_or_context_feedback(tmp_path):
    catalog, engine, key, slot, frozen = fixture_trial(tmp_path)
    before = catalog.latest(key)
    assert frozen["predicted_ids"] == ["squat"]
    context = catalog.context()
    context["active_plan"] = ["bench_press"]
    catalog.save_context(context)
    assert engine.capture(key, slot["slot_id"]) == frozen
    label = engine.label(frozen["trial_id"], None, {"exercise_id": "squat"})
    assert engine.frozen(frozen["trial_id"]) == frozen
    assert catalog.latest(key) == before
    assert catalog.context() == context
    assert engine.metrics()["top3_hits"] == 1
    assert engine.metrics()["cohorts"]["local_real"]["labelled_exercise_blocks"] == 0
    assert engine.metrics()["cohorts"]["local_real"]["top3_hit_fraction_all_labelled"] is None
    with pytest.raises(DataError):
        engine.label(frozen["trial_id"], None, {})
    engine.label(frozen["trial_id"], label["revision_id"], {})
    assert engine.metrics()["labelled_exercise_blocks"] == 0
    assert engine.metrics()["unknown_exercise_blocks"] == 1
    assert len(list((engine.root / frozen["trial_id"] / "labels").glob("*.json"))) == 2


def test_no_evidence_abstains_and_labels_remain_independent(tmp_path):
    catalog, engine, key, slot, _ = fixture_trial(tmp_path)
    # A second generated source has no device hints and different motion samples.
    other = catalog.import_fit(strength_fit_bytes(device_sets=False), synthetic=True)
    context = catalog.context()
    context["active_plan"] = []
    catalog.save_context(context)
    work = next(s for s in catalog.latest(other)["slots"] if s["kind"] == "work_candidate")
    frozen = engine.capture(other, work["slot_id"])
    assert frozen["suggestions"]["abstained"]
    engine.label(
        frozen["trial_id"], None, {"exercise_id": "squat", "movement_families": ["tirata"]}
    )
    metrics = engine.metrics()
    assert metrics["top3_misses_including_abstention"] == 1
    assert metrics["labelled_nonabstained_blocks"] == 0
    assert metrics["top3_hit_fraction_nonabstained_labelled"] is None
    assert metrics["labelled_family_blocks"] == 1 and metrics["top3_family_hits"] == 0
    # No family is inferred from the exercise declaration.
    assert engine.latest_label(frozen["trial_id"])["values"]["movement_families"] == ["tirata"]
    catalog.edit(
        key,
        catalog.latest(key)["revision_id"],
        {"type": "assign", "slot_id": slot["slot_id"], "exercise_id": "squat"},
    )
    # Existing first capture remains immutable even after subsequent reconciliation.
    assert engine.capture(key, slot["slot_id"])["predicted_ids"] == ["squat"]


def test_assigned_target_cannot_create_new_capture(tmp_path):
    catalog = StrengthCatalog(tmp_path)
    key = catalog.import_fit(strength_fit_bytes(), synthetic=True)
    slot = next(s for s in catalog.latest(key)["slots"] if s["kind"] == "work_candidate")
    catalog.edit(
        key,
        catalog.latest(key)["revision_id"],
        {"type": "assign", "slot_id": slot["slot_id"], "exercise_id": "squat"},
    )
    with pytest.raises(DataError):
        StrengthEvaluation(tmp_path).capture(key, slot["slot_id"])


@pytest.mark.parametrize(
    "values",
    [
        {"exercise_id": "not-in-catalog"},
        {"exercise_id": []},
        {"movement_families": [["spinta"]]},
        {"movement_families": ["anca", "anca"]},
        {"usefulness": []},
        {"corrections": -1},
        {"corrections": True},
        {"extra": 1},
    ],
)
def test_invalid_labels_do_not_write(tmp_path, values):
    _, engine, _, _, frozen = fixture_trial(tmp_path)
    with pytest.raises(DataError):
        engine.label(frozen["trial_id"], None, values)
    assert engine.latest_label(frozen["trial_id"]) is None


def test_tampering_detected_and_paths_rejected(tmp_path):
    _, engine, _, _, frozen = fixture_trial(tmp_path)
    with pytest.raises(DataError):
        engine.frozen("../../uv.lock")
    path = engine.root / frozen["trial_id"] / "frozen.json"
    row = json.loads(path.read_text())
    row["predicted_ids"] = []
    path.write_text(json.dumps(row))
    with pytest.raises(DataError, match="Snapshot"):
        engine.metrics()


def test_explicit_denominators_empty_unknown_and_abstention():
    empty = evaluation_metrics([])
    assert empty["suggestion_coverage_fraction"] is None
    assert empty["top3_hit_fraction_all_labelled"] is None
    base = {
        "synthetic": True,
        "session_id": "x",
        "suggestions": {"abstained": False},
        "predicted_ids": ["squat"],
        "predicted_families": ["anca"],
        "garmin_ids": [],
        "context_ids": ["squat"],
        "label": {
            "exercise_id": "squat",
            "movement_families": ["anca"],
            "corrections": 0,
            "usefulness": "helpful",
        },
    }
    miss = copy.deepcopy(base)
    miss.update(suggestions={"abstained": True}, predicted_ids=[], predicted_families=[])
    unknown = copy.deepcopy(base)
    unknown["label"] = {}
    rows = [base, miss, unknown]
    before = copy.deepcopy(rows)
    result = evaluation_metrics(rows)
    assert rows == before
    assert result["suggestion_coverage_fraction"] == 2 / 3
    assert result["labelled_exercise_blocks"] == 2
    assert result["top3_hit_fraction_all_labelled"] == 0.5
    assert result["top3_hit_fraction_nonabstained_labelled"] == 1
    assert result["top3_family_hit_fraction"] == 0.5
    assert result["context_baseline_hits"] == 2 and result["garmin_baseline_hits"] == 0
    assert result["correction_observations"] == 2 and result["declared_corrections_mean"] == 0


def test_ux_latest_opt_in_only_and_unknown_not_zero(tmp_path):
    sessions = StrengthSessions(tmp_path)
    sessions.save(None, None, {}, synthetic=True)
    assert sessions.ux_metrics()["annotation_elapsed_mean_s"] is None


def test_local_report_commands_reproduce_and_preserve(tmp_path):
    from hybrid_training_analytics.cli import main
    from hybrid_training_analytics.extended_cli import descriptor_output
    from hybrid_training_analytics.local_catalog import LocalCatalog

    _, engine, key, _, frozen = fixture_trial(tmp_path / "workspace")
    engine.label(frozen["trial_id"], None, {"exercise_id": "squat"})
    StrengthSessions(tmp_path / "workspace").save("fit-" + key, None, {}, synthetic=True)
    for command in ("local-hybrid-report", "strength-evaluation-report"):
        for name in ("a", "b"):
            output = tmp_path / (command + name)
            assert (
                main([command, "--workspace", str(tmp_path / "workspace"), "--out", str(output)])
                == 0
            )
        for filename in ("analysis.json", "report.html", "manifest.json"):
            assert (tmp_path / (command + "a") / filename).read_bytes() == (
                tmp_path / (command + "b") / filename
            ).read_bytes()
        assert (
            main(
                [
                    command,
                    "--workspace",
                    str(tmp_path / "workspace"),
                    "--out",
                    str(tmp_path / (command + "a")),
                ]
            )
            != 0
        )
    payload = LocalCatalog(tmp_path / "workspace").hybrid()
    payload["strength_sessions"][0]["body_scope"] = '<script>alert("x")</script>'
    descriptor_output(payload, tmp_path / "escaped")
    html = (tmp_path / "escaped" / "report.html").read_text(encoding="utf-8")
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_capture_excludes_target_adjacency_and_keeps_native_baseline(tmp_path):
    catalog = StrengthCatalog(tmp_path)
    key = catalog.import_fit(strength_fit_bytes(), synthetic=True)
    context = catalog.context()
    context["active_plan"] = ["bench_press"]
    catalog.save_context(context)
    work = [s for s in catalog.latest(key)["slots"] if s["kind"] == "work_candidate"]
    for op in (
        {"type": "assign", "slot_id": work[0]["slot_id"], "exercise_id": "bench_press"},
        {
            "type": "confirm",
            "slot_id": work[0]["slot_id"],
            "values": {"reps": 8, "external_load_kg": 40},
        },
    ):
        catalog.edit(key, catalog.latest(key)["revision_id"], op)
    engine = StrengthEvaluation(tmp_path)
    frozen = engine.capture(key, work[1]["slot_id"])
    assert frozen["suggestions"]["history_refs"] == []
    components = frozen["suggestions"]["top3"][0]["components"]
    assert components["adjacent_muscle"] == 0
    noisy = catalog.import_fit(
        strength_fit_bytes(acceleration=False, noisy=True, category_codes=[0, 28]), synthetic=True
    )
    first = next(s for s in catalog.latest(noisy)["slots"] if s["start_s"] == 0)
    native = engine.capture(noisy, first["slot_id"])
    assert set(native["garmin_ids"]) == {"bench_press", "squat"}
    assert all(not c["eligible_for_ranking"] for c in native["suggestions"]["native_candidates"])


def test_hybrid_report_reproducible_across_process_hash_seeds(tmp_path):
    import os
    import subprocess
    import sys

    StrengthSessions(tmp_path / "workspace").save(None, None, {}, synthetic=True)
    for seed in ("1", "17"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "hybrid_training_analytics",
                "local-hybrid-report",
                "--workspace",
                str(tmp_path / "workspace"),
                "--out",
                str(tmp_path / seed),
            ],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    for name in ("analysis.json", "report.html", "manifest.json"):
        assert (tmp_path / "1" / name).read_bytes() == (tmp_path / "17" / name).read_bytes()
