import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from hybrid_training_analytics.advanced_metrics import (
    DecouplingConfig,
    channels,
    compare_phases,
    decoupling,
    minute_bins,
    sensitivity,
    support,
)
from hybrid_training_analytics.annotations import append_revision, validate_annotations
from hybrid_training_analytics.hybrid import hybrid_summary, session_load
from hybrid_training_analytics.ingestion import read_fit
from hybrid_training_analytics.models import DataError, RunData, Sample
from hybrid_training_analytics.signals import read_signals, validate_signals
from hybrid_training_analytics.strength import StrengthSet
from hybrid_training_analytics.synthetic import fit_bytes


def invented(times=range(0, 1201, 5), hr=lambda t: 140 + 0.01 * t + math.sin(t / 10)):
    origin = datetime(2025, 1, 1, tzinfo=UTC)
    return RunData(
        [
            Sample((origin + timedelta(seconds=t)).isoformat(), float(t), hr(t), 3.0, t * 3.0)
            for t in times
        ],
        "a" * 64,
        synthetic=True,
    )


def eligible(**kwargs):
    return DecouplingConfig(start_s=0, output_confirmed=True, constant_incline=True, **kwargs)


def test_irregular_minutes_and_partial():
    run = invented([0, 10, 40, 60, 75], hr=lambda t: 100 + t)
    bins = minute_bins(run, max_gap_s=40)
    assert bins[0]["heart_rate_bpm"] == pytest.approx(130)
    assert bins[0]["heart_rate_bpm_valid_s"] == 60
    assert bins[1]["span_s"] == 15
    assert bins[1]["heart_rate_bpm"] == pytest.approx(167.5)
    assert run.samples[1].heart_rate_bpm == 110


def test_missing_pause_and_gap_not_filled():
    run = invented([0, 5, 10, 15, 20, 60])
    run.samples[2] = replace(run.samples[2], heart_rate_bpm=None)
    run.pauses_s = [(15, 20)]
    row = minute_bins(run)[0]
    assert row["heart_rate_bpm_valid_s"] == 5
    assert row["speed_m_s_valid_s"] == 15


def test_common_support_not_independent_means():
    run = invented([0, 5, 10, 15, 20], hr=lambda t: 100 + t)
    run.samples[1] = replace(run.samples[1], speed_m_s=None)
    run.samples[4] = replace(run.samples[4], heart_rate_bpm=None)
    value = support(run, channels(run), ("heart_rate_bpm", "speed_m_s"), 0, 20)
    assert value["valid_s"] == 5
    assert value["means"]["heart_rate_bpm"] == 112.5


def test_decoupling_known_formula_and_scaling_cancels():
    run = invented()
    first = decoupling(run, eligible())
    halves = first["halves"]
    expected = 100 * (
        1 - halves[0]["means"]["heart_rate_bpm"] / halves[1]["means"]["heart_rate_bpm"]
    )
    assert first["status"] == "available"
    assert first["decoupling_pct"] == pytest.approx(expected)
    assert decoupling(run, eligible(), factor=1.3)["decoupling_pct"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("pause", "timer_pause"),
        ("gap", "gap"),
        ("hr", "suspect_hr_jump"),
        ("frozen", "constant_hr_120s"),
        ("speed", "variable_output"),
        ("missing", "insufficient_joint_coverage"),
    ],
)
def test_decoupling_exclusion(fault, reason):
    run = invented()
    if fault == "pause":
        run.pauses_s = [(300, 305)]
    elif fault == "gap":
        run.samples = [s for s in run.samples if not 300 < s.elapsed_s < 330]
    elif fault == "hr":
        run.samples[80] = replace(run.samples[80], heart_rate_bpm=200)
    elif fault == "frozen":
        run = invented(hr=lambda t: 140)
    elif fault == "speed":
        run.samples[80] = replace(run.samples[80], speed_m_s=4)
    else:
        run.samples = [
            replace(s, heart_rate_bpm=None) if s.elapsed_s < 200 else s for s in run.samples
        ]
    value = decoupling(run, eligible())
    assert value["decoupling_pct"] is None
    assert reason in value["reasons"]


def test_confirmations_and_sensitivity():
    run = invented()
    assert "output_not_confirmed_by_user" in decoupling(run)["reasons"]
    variants = sensitivity(run, eligible())
    assert [v["window_s"] for v in variants] == [[0, 1200], [60, 1200], [0, 1140]]
    assert all(v["status"] == "available" for v in variants)


def test_signal_crc_hash_alignment_and_nulls(tmp_path):
    path = tmp_path / "synthetic.fit"
    run = invented()
    path.write_bytes(
        fit_bytes(
            run.samples, signals=[{"cadence_fit_rpm": 85, "power_w": 200} for _ in run.samples]
        )
    )
    run = read_fit(path, synthetic=True)
    payload = read_signals(path, run)
    assert payload["samples"][1]["cadence_fit_rpm"] == 85
    assert payload["samples"][1]["power_w"] == 200
    assert decoupling(run, eligible(output="power_w"), payload)["status"] == "available"
    payload["samples"][1]["timestamp_utc"] = "2000-01-01T00:00:00+00:00"
    with pytest.raises(DataError):
        validate_signals(run, payload)
    with pytest.raises(DataError):
        read_signals(path, invented())
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(DataError):
        read_signals(path, run)


def test_variable_cadence_blocks():
    run = invented()
    payload = {
        "schema_version": 1,
        "artifact_type": "fit_signals",
        "source_sha256": run.source_sha256,
        "samples": [
            {
                "timestamp_utc": s.timestamp_utc,
                "cadence_fit_rpm": 85 if i != 50 else 70,
                "power_w": None,
            }
            for i, s in enumerate(run.samples)
        ],
    }
    assert "variable_cadence" in decoupling(run, eligible(), payload)["reasons"]


@pytest.mark.parametrize(
    "payload",
    [
        {"sensor": "fabricated"},
        {"session_rpe": float("nan")},
        {"speed_confirmed": "true"},
        {"target_distance_m": 5000},
        {"start_utc": "2025-01-01"},
        {"unknown": 1},
        {"incline_pct": False},
    ],
)
def test_annotations_invalid(payload):
    with pytest.raises(DataError):
        validate_annotations(payload)


def test_revision_append_only(tmp_path):
    first = append_revision(tmp_path, "a" * 64, {"sensor": "wrist"})
    original = (tmp_path / (first["revision_id"] + ".json")).read_bytes()
    second = append_revision(tmp_path, "a" * 64, {"sensor": "chest"})
    assert first["revision_id"] != second["revision_id"]
    assert (tmp_path / (first["revision_id"] + ".json")).read_bytes() == original
    assert first["values"]["session_rpe"] is None


def test_load_and_concurrent_context_no_cause():
    sets = [
        StrengthSet("2025-01-01T12:10:00+00:00", "s", "squat", "external", 80, 0, 50, 1, 5, 2, 8)
    ]
    annotation = {
        "s": {
            "start_utc": "2025-01-01T12:00:00+00:00",
            "end_utc": "2025-01-01T13:00:00+00:00",
            "session_rpe": 8,
            "lower_body_exposure": "high",
            "technique": "standard",
        }
    }
    summary = hybrid_summary(
        sets,
        annotation,
        [{"run_id": "r", "start_utc": "2025-01-02T13:00:00+00:00", "duration_s": 1800}],
    )
    assert summary["strength_sessions"][0]["load"]["session_rpe_load_au"] == 480
    assert summary["strength_sessions"][0]["exercises"][0]["external_volume_kg_reps"] == 250
    context = summary["run_contexts"][0]["context"]
    assert context["preceding_strength"][0]["hours_since_end"] == 24
    assert context["interference_score"] is None
    assert summary["weeks"]["2025-W01"]["rpe_load_complete"] is False
    assert session_load({})["session_rpe_load_au"] is None


def test_matching_strength_conditions():
    sets = [
        StrengthSet(
            f"2025-01-0{day}T12:10:00+00:00",
            f"s{day}",
            "squat",
            "external",
            80,
            0,
            load,
            1,
            5,
            2,
            8,
        )
        for day, load in [(1, 50), (2, 55)]
    ]
    summary = hybrid_summary(sets, {"s1": {"technique": "a"}, "s2": {"technique": "a"}}, [])
    assert summary["strength_trends"][0]["observations"][1]["delta_load_kg"] == 5
    summary = hybrid_summary(sets, {"s1": {"technique": "a"}, "s2": {"technique": "b"}}, [])
    assert all(len(row["observations"]) == 1 for row in summary["strength_trends"])


def test_comparison_requires_matching_context():
    a = decoupling(invented(), eligible())
    context = {"sensor": "wrist", "incline_pct": 1, "output_basis": "device_estimate_confirmed"}
    assert (
        compare_phases(a, a, context_a=context, context_b=context)[
            "delta_decoupling_percentage_points"
        ]
        == 0
    )
    result = compare_phases(a, a, context_a=context, context_b={**context, "sensor": "chest"})
    assert result["status"] == "unavailable"
