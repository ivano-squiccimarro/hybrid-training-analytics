from pathlib import Path

import pytest

from hybrid_training_analytics.models import DataError
from hybrid_training_analytics.reconciliation import detect_slots
from hybrid_training_analytics.strength_fit import read_strength_fit
from hybrid_training_analytics.strength_synthetic import strength_fit_bytes


def parse(tmp_path: Path, **kwargs):
    path = tmp_path / "synthetic.fit"
    path.write_bytes(strength_fit_bytes(**kwargs))
    return read_strength_fit(path, synthetic=True)


def test_strength_parser_units_offsets_and_device_annotations(tmp_path):
    raw = parse(tmp_path)
    assert raw["start_utc"] == "2025-02-01T12:00:00+00:00"
    assert raw["duration_s"] == 300
    assert len(raw["accelerometer"]) == 3000
    assert raw["accelerometer"][11]["elapsed_s"] == 1.1
    assert raw["accelerometer"][11]["accel_z_g"] == 1
    assert raw["device_sets"][0]["weight_device_kg"] == 50
    assert raw["device_sets"][0]["repetitions_device"] == 10
    assert raw["device_sets"][0]["end_s"] == 60


def test_detection_recovers_missing_set_without_laps(tmp_path):
    raw = parse(tmp_path)
    draft = detect_slots(raw)
    work = [s for s in draft["slots"] if s["kind"] == "work_candidate"]
    assert [(s["start_s"], s["end_s"]) for s in work] == [(30, 60), (180, 210)]
    assert work[0]["device_reps"] == 10
    assert work[1]["device_reps"] is None
    assert all(s["reps"] is None and not s["confirmed"] for s in work)
    assert draft["detection_mode"] == "motion_clusters"


@pytest.mark.parametrize(
    "options,mode",
    [
        ({"acceleration": False}, "device_hints"),
        ({"acceleration": False, "device_sets": False}, "unresolved"),
        ({"missing_offsets": True}, "device_hints"),
    ],
)
def test_missing_motion_no_hr_only_set_invention(tmp_path, options, mode):
    draft = detect_slots(parse(tmp_path, **options))
    assert draft["detection_mode"] == mode
    assert all(f["motion_rms"] is None for f in draft["features"])
    assert all(s["provenance"] == "algorithm_hypothesis" for s in draft["slots"])
    if mode == "unresolved":
        assert [(s["start_s"], s["end_s"], s["kind"]) for s in draft["slots"]] == [
            (0, 300, "unknown")
        ]


def test_counts_remain_counts_not_fake_calibrated_g(tmp_path):
    raw = parse(tmp_path, calibrated=False)
    assert raw["accelerometer"][0]["accel_z_g"] is None
    assert raw["accelerometer"][0]["accel_z_counts"] == 33768
    draft = detect_slots(raw)
    assert draft["features"][10]["motion_units"] == "counts"
    assert draft["detection_mode"] == "motion_clusters"


def test_crc_truncation_and_multisession_rejected(tmp_path):
    path = tmp_path / "bad.fit"
    content = strength_fit_bytes()
    for bad in (content[:-8], content + content, content[:-1] + bytes([content[-1] ^ 1])):
        path.write_bytes(bad)
        with pytest.raises(DataError):
            read_strength_fit(path)
