from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from hybrid_training_analytics.calibration import CalibrationConfig, calibrate, running_cost_j_kg_m
from hybrid_training_analytics.models import DataError, RunData, Sample


def test_uniform_scaling_conserves_declared_total_and_source(run):
    original = run.to_dict()
    result = calibrate(run, CalibrationConfig(10800, "watch_final"))
    assert result["scale_factor"] == 2
    assert result["samples"][0]["scaled_distance_m"] == 0
    assert result["samples"][-1]["scaled_distance_m"] == 10800
    assert result["samples"][1]["scaled_record_speed_m_s"] == 6
    assert result["samples"][1]["scaled_record_pace_estimate_s_km"] == pytest.approx(1000 / 6)
    assert result["samples"][1]["scaled_interval_distance_m"] == 30
    assert run.to_dict() == original
    assert result["source_sha256"] == run.source_sha256
    assert result["samples"][100]["heart_rate_bpm"] is None
    assert result["gap_model"]["status"] == "unavailable"


def test_irregular_intervals_nulls_zero_and_pauses_are_preserved():
    origin = datetime(2025, 1, 1, tzinfo=UTC)
    times = [0, 2, 5, 7, 10, 12, 25]
    distances = [10, 14, None, 24, 24, 30, 40]
    speeds = [2, 2, None, 2, 0, 3, 3]
    run = RunData(
        [
            Sample((origin + timedelta(seconds=t)).isoformat(), t, None, speed, distance)
            for t, speed, distance in zip(times, speeds, distances, strict=True)
        ],
        "a" * 64,
        synthetic=True,
        pauses_s=[(6, 9)],
    )
    result = calibrate(run, CalibrationConfig(60, "user_other", incline_pct=0))
    samples = result["samples"]
    assert [s["elapsed_s"] for s in samples] == times
    assert [s["scaled_distance_m"] for s in samples] == [0, 8, None, 28, 28, 40, 60]
    assert samples[0]["scaled_interval_speed_estimate_m_s"] is None
    assert samples[1]["scaled_interval_speed_estimate_m_s"] == 4  # 8 m / 2 s
    assert samples[2]["scaled_record_speed_m_s"] is None
    assert samples[2]["scaled_record_pace_estimate_s_km"] is None
    assert samples[2]["interval_status"] == "missing_distance"
    assert samples[3]["scaled_record_speed_m_s"] == 4  # preserved device channel during pause
    assert samples[3]["scaled_record_pace_estimate_s_km"] is None
    assert samples[3]["gap_model_estimate_s_km"] is None
    assert samples[3]["interval_status"] == "timer_pause"
    assert samples[4]["timer_active"] is True
    assert samples[4]["scaled_record_pace_estimate_s_km"] is None  # zero speed
    assert samples[4]["scaled_interval_speed_estimate_m_s"] is None  # crossing pause
    assert samples[5]["scaled_interval_speed_estimate_m_s"] == 6  # 12 m / 2 s
    assert samples[6]["interval_status"] == "gap_exceeds_limit"
    assert samples[6]["scaled_interval_distance_m"] is None
    assert result["pauses_s"] == [(6, 9)]


def test_record_speed_missing_is_not_filled_from_distance(run):
    run.samples[1] = replace(run.samples[1], speed_m_s=None)
    sample = calibrate(run, CalibrationConfig(5400, "watch_final"))["samples"][1]
    assert sample["scaled_record_speed_m_s"] is None
    assert sample["scaled_record_pace_estimate_s_km"] is None
    assert sample["scaled_interval_speed_estimate_m_s"] == 3  # separately labelled estimate


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "5400"])
def test_invalid_declared_distance_rejected(run, value):
    with pytest.raises(DataError):
        calibrate(run, CalibrationConfig(value, "watch_final"))


@pytest.mark.parametrize("endpoint", ["first", "last", "zero_delta"])
def test_no_endpoint_imputation_or_zero_division(run, endpoint):
    if endpoint == "first":
        run.samples[0] = replace(run.samples[0], distance_m=None)
    elif endpoint == "last":
        run.samples[-1] = replace(run.samples[-1], distance_m=None)
    else:
        run.samples = [replace(s, distance_m=10) for s in run.samples]
    with pytest.raises(DataError, match="endpoint"):
        calibrate(run, CalibrationConfig(5400, "watch_final"))


def test_gap_model_level_and_known_uphill_polynomial(run):
    assert running_cost_j_kg_m(0) == 3.6
    assert running_cost_j_kg_m(10) == pytest.approx(5.968214)
    flat = calibrate(run, CalibrationConfig(5400, "watch_final", 0))["samples"][0]
    uphill = calibrate(run, CalibrationConfig(5400, "watch_final", 10))["samples"][0]
    assert flat["gap_model_estimate_s_km"] == flat["scaled_record_pace_estimate_s_km"]
    assert uphill["gap_model_estimate_s_km"] == pytest.approx((1000 / 3) * 3.6 / 5.968214)
    assert uphill["gap_model_estimate_s_km"] < flat["gap_model_estimate_s_km"]


@pytest.mark.parametrize("incline", [21, -21, float("nan"), True, "1"])
def test_invalid_grade_rejected_without_clamping(run, incline):
    with pytest.raises(DataError):
        calibrate(run, CalibrationConfig(5400, "watch_final", incline))


def test_source_and_config_not_implicitly_confirmed(run):
    with pytest.raises(DataError):
        calibrate(run, CalibrationConfig(5400, "unknown"))
    with pytest.raises(DataError):
        calibrate(run, CalibrationConfig(5400, "watch_final", max_gap_s=0))
    from hybrid_training_analytics.metrics import analyze

    calibrate(run, CalibrationConfig(5400, "watch_final", 1))
    analysis = analyze(run)
    assert analysis["drift"]["status"] == "unavailable"
    assert analysis["estimates"]["pace_s_km_user_confirmed_speed"] is None
