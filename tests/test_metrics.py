from dataclasses import replace

import pytest

from hybrid_training_analytics.metrics import (
    AnalysisConfig,
    TreadmillSegment,
    analyze,
    read_treadmill,
    weighted_mean,
)
from hybrid_training_analytics.models import DataError


def test_unverified_speed_cannot_produce_drift_or_pace(run):
    result = analyze(run)
    assert result["drift"]["status"] == "unavailable"
    assert result["estimates"]["pace_s_km_user_confirmed_speed"] is None


def test_constant_verified_speed_and_small_missing_hr(run):
    result = analyze(run, AnalysisConfig(speed_reliable=True, constant_incline=True))
    assert result["drift"]["status"] == "available"
    assert 2 < result["drift"]["heart_rate_change_pct"] < 8
    assert result["quality"]["missing_hr_records"] == 2
    assert result["estimates"]["pace_s_km_user_confirmed_speed"] == pytest.approx(1000 / 3)


def test_treadmill_comparison_never_corrects_garmin(run):
    result = analyze(run, treadmill=[TreadmillSegment(0, 1800, 12, 1)])
    assert result["comparison"]["difference_pct"] == pytest.approx(-10)
    assert result["drift"]["status"] == "available"
    assert run.samples[-1].distance_m == 5400


def test_treadmill_partial_coverage_not_compared(run):
    result = analyze(run, treadmill=[TreadmillSegment(300, 1500, 12, 1)])
    assert result["comparison"]["status"] == "unavailable"


def test_variable_speed_refuses_drift(run):
    run.samples[200] = replace(run.samples[200], speed_m_s=4)
    result = analyze(run, AnalysisConfig(speed_reliable=True, constant_incline=True))
    assert result["drift"]["status"] == "unavailable"


@pytest.mark.parametrize("fault", ["gap", "spike", "missing", "pause"])
def test_drift_rejects_bad_signal_or_pauses(run, fault):
    if fault == "gap":
        run.samples = [s for s in run.samples if not 500 < s.elapsed_s < 700]
    elif fault == "spike":
        run.samples[200] = replace(run.samples[200], heart_rate_bpm=220)
    elif fault == "missing":
        run.samples = [
            replace(s, heart_rate_bpm=None) if s.elapsed_s > 1200 else s for s in run.samples
        ]
    else:
        run.pauses_s = [(700, 720)]
    config = AnalysisConfig(speed_reliable=True, constant_incline=True)
    assert analyze(run, config)["drift"]["status"] == "unavailable"


def test_weighted_average_is_time_weighted_and_gap_aware(run):
    run.samples = [
        replace(run.samples[i], heart_rate_bpm=value) for i, value in [(0, 100), (1, 110), (3, 130)]
    ]
    mean, duration = weighted_mean(run, "heart_rate_bpm", 0, 15, 10)
    assert duration == 15
    assert mean == pytest.approx((105 * 5 + 120 * 10) / 15)
    _, duration_short = weighted_mean(run, "heart_rate_bpm", 0, 15, 6)
    assert duration_short == 5


def test_invalid_rpe_and_window_rejected(run):
    with pytest.raises(DataError):
        analyze(run, AnalysisConfig(rpe=float("nan")))
    with pytest.raises(DataError):
        analyze(run, AnalysisConfig(segment_end_s=2000))


def test_treadmill_overlap_rejected(tmp_path):
    path = tmp_path / "treadmill.csv"
    path.write_text("start_s,end_s,speed_kmh,incline_pct\n0,800,12,1\n700,1800,12,1\n")
    with pytest.raises(DataError, match="sovrapposti"):
        read_treadmill(path)


def test_speed_confirmation_alone_does_not_confirm_incline(run):
    result = analyze(run, AnalysisConfig(speed_reliable=True))
    assert result["drift"]["status"] == "unavailable"


def test_invalid_restored_config_rejected(run):
    with pytest.raises(DataError):
        analyze(run, AnalysisConfig(warmup_s="300"))
    with pytest.raises(DataError):
        analyze(run, treadmill=[TreadmillSegment(0, 1800, 0, 1)])


@pytest.mark.parametrize("interruption", ["continuous", "pause", "null", "gap"])
def test_constant_hr_requires_uninterrupted_active_signal(run, interruption):
    run.samples = [
        replace(s, heart_rate_bpm=100 if s.elapsed_s <= 150 else 130 + s.elapsed_s / 180)
        for s in run.samples
    ]
    if interruption == "pause":
        run.pauses_s = [(70, 80)]
    elif interruption == "null":
        run.samples = [
            replace(s, heart_rate_bpm=None) if s.elapsed_s == 75 else s for s in run.samples
        ]
    elif interruption == "gap":
        run.samples = [s for s in run.samples if not 70 < s.elapsed_s < 90]
    config = AnalysisConfig(speed_reliable=True, constant_incline=True)
    result = analyze(run, config)
    expected_duration_s = 150 if interruption == "continuous" else 70
    assert result["quality"]["longest_constant_hr_s"] == expected_duration_s
    # The interruption is before warmup: the later valid window remains eligible.
    assert result["drift"]["status"] == (
        "unavailable" if interruption == "continuous" else "available"
    )
    assert run.samples[0].heart_rate_bpm == 100
