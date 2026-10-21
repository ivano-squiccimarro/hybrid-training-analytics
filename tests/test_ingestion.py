from datetime import UTC, datetime, timedelta

import pytest

from hybrid_training_analytics.ingestion import normalize_records, read_fit
from hybrid_training_analytics.models import DataError, RunData, utc_timestamp
from hybrid_training_analytics.synthetic import demo_samples, fit_bytes


def test_real_decoder_reads_synthetic_binary(synthetic_fit):
    run = read_fit(synthetic_fit)
    assert len(run.samples) == 361
    assert run.samples[-1].elapsed_s == 1800
    assert run.samples[-1].distance_m == 5400
    assert run.samples[0].speed_m_s == 3
    assert run.samples[100].heart_rate_bpm is None
    assert run.samples[0].timestamp_utc.endswith("+00:00")


def test_enhanced_speed_is_decoded(tmp_path):
    path = tmp_path / "enhanced.fit"
    path.write_bytes(fit_bytes(demo_samples(), enhanced_speed=True))
    assert read_fit(path).samples[0].speed_m_s == 3


def test_crc_failure_is_not_partial_success(synthetic_fit):
    data = bytearray(synthetic_fit.read_bytes())
    data[-1] ^= 1
    synthetic_fit.write_bytes(data)
    with pytest.raises(DataError, match="CRC"):
        read_fit(synthetic_fit)


@pytest.mark.parametrize(
    "content",
    [b"not FIT", b"", fit_bytes(demo_samples())[:-20]],
    ids=["invalid", "empty", "truncated"],
)
def test_truncated_or_invalid_fit(tmp_path, content):
    path = tmp_path / "broken.fit"
    path.write_bytes(content)
    with pytest.raises(DataError):
        read_fit(path)


def test_order_duplicates_missing_values_and_distance_resets():
    start = datetime(2025, 1, 1, tzinfo=UTC)
    records = [
        {"timestamp": start + timedelta(seconds=5), "heart_rate": 0, "distance": 10},
        {"timestamp": start, "heart_rate": 120, "distance": 0},
        {"timestamp": start + timedelta(seconds=5), "heart_rate": 140, "distance": 10},
        {
            "timestamp": start + timedelta(seconds=10),
            "heart_rate": 0,
            "distance": 5,
            "speed": float("nan"),
        },
        {"heart_rate": 120},
    ]
    run = normalize_records(records, digest="a" * 64)
    assert len(run.samples) == 3
    assert run.samples[1].heart_rate_bpm == 140
    assert run.samples[-1].heart_rate_bpm is None
    assert run.samples[-1].distance_m is None
    assert len(run.warnings) >= 4


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        ([(600, "stop_all"), (660, "start")], [(600, 660)]),
        ([(600, "stop_all"), (600, "start")], []),
        ([(600, "stop"), (600, "start"), (900, "stop_all"), (960, "start")], [(900, 960)]),
        ([(0, "start"), (1500, "stop_all")], [(1500, 1800)]),
    ],
    ids=["pause", "same-second", "same-second-then-pause", "open-stop-until-end"],
)
def test_timer_events_become_bounded_pauses(events, expected):
    start = datetime(2025, 1, 1, tzinfo=UTC)
    records = [{"timestamp": start + timedelta(seconds=s), "heart_rate": 140} for s in range(1801)]
    stamped = [(utc_timestamp(start + timedelta(seconds=s), fit_utc=True), k) for s, k in events]
    run = normalize_records(records, digest="a" * 64, events=stamped)
    assert run.pauses_s == expected


def test_normalized_json_rejects_naive_timestamp(run):
    payload = run.to_dict()
    payload["samples"][0]["timestamp_utc"] = "2025-01-15T17:00:00"
    with pytest.raises(DataError, match="offset"):
        RunData.from_dict(payload)


def test_timezone_dst_is_not_host_timezone(run):
    from zoneinfo import ZoneInfo

    utc = datetime.fromisoformat(run.samples[0].timestamp_utc)
    assert utc.astimezone(ZoneInfo("Europe/Rome")).hour == 18
    assert datetime(2025, 7, 1, tzinfo=UTC).astimezone(ZoneInfo("Europe/Rome")).hour == 2
    assert utc_timestamp("2025-10-26T02:30:00+02:00") != utc_timestamp("2025-10-26T02:30:00+01:00")


def test_multisport_or_chained_rejected(tmp_path):
    path = tmp_path / "chained.fit"
    path.write_bytes(fit_bytes(demo_samples()) * 2)
    with pytest.raises(DataError, match="multisport"):
        read_fit(path)
