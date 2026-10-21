"""Synthetic quality, ZIP and candidate normalization regression tests."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from hybrid_training_analytics.exercise_vocabulary import normalize_categories
from hybrid_training_analytics.models import DataError
from hybrid_training_analytics.reconciliation import detect_slots
from hybrid_training_analytics.strength_catalog import StrengthCatalog
from hybrid_training_analytics.strength_fit import read_strength_fit
from hybrid_training_analytics.strength_input import MAX_INPUT_BYTES, unpack_strength_input
from hybrid_training_analytics.strength_quality import QualityConfig, assess_strength_quality
from hybrid_training_analytics.strength_synthetic import strength_fit_bytes
from hybrid_training_analytics.strength_workflow import strength_output, write_strength_output


def raw_fixture(tmp_path, **kwargs):
    path = tmp_path / "synthetic.fit"
    path.write_bytes(strength_fit_bytes(**kwargs))
    return read_strength_fit(path, synthetic=True)


def zip_fixture(content, name="invented.fit", compression=ZIP_DEFLATED):
    stream = io.BytesIO()
    with ZipFile(stream, "w", compression=compression) as archive:
        archive.writestr(name, content)
    return stream.getvalue()


def codes(quality, section):
    return {row["code"] for row in quality["sections"][section]}


def test_quality_clean_coverage_and_immutable_raw(tmp_path):
    raw = raw_fixture(tmp_path)
    original = copy.deepcopy(raw)
    quality = assess_strength_quality(raw)
    assert quality["hr"]["supported_duration_s"] == 300
    assert quality["hr"]["temporal_coverage_fraction"] == 1
    assert quality["accelerometer"]["supported_duration_s"] == pytest.approx(299.9)
    assert quality["accelerometer"]["counts_samples"] == 0
    assert {"session_time", "hr_present", "movement_recorded"} <= codes(quality, "available")
    assert quality["duration_checks"][0]["upper_limit_s"] is None
    assert raw == original
    assert not any(s["confirmed"] for s in detect_slots(raw)["slots"])


def test_null_hr_breaks_support_gap_and_outside_preserved(tmp_path):
    raw = raw_fixture(tmp_path, acceleration=False)
    raw["records"] = [
        {"elapsed_s": t, "heart_rate_bpm": hr}
        for t, hr in [(-5, 110), (0, 110), (5, None), (10, 115), (25, 120), (30, 121), (310, 122)]
    ]
    quality = assess_strength_quality(raw)
    assert quality["hr"]["supported_duration_s"] == 5
    assert quality["hr"]["valid_in_bounds"] == 4
    assert quality["hr"]["large_gap_count"] == 1
    assert "movement_unavailable" in codes(quality, "undetermined")
    assert len(raw["records"]) == 7 and raw["records"][1]["heart_rate_bpm"] == 110


def test_incomplete_accel_not_usable_and_unknown_not_motion(tmp_path):
    raw = raw_fixture(tmp_path, missing_offsets=True, device_sets=False)
    raw["message_counts"]["unknown_999"] = 7
    quality = assess_strength_quality(raw)
    assert quality["accelerometer"]["usable_in_bounds"] == 0
    assert quality["accelerometer"]["supported_duration_s"] == 0
    assert quality["uninterpreted_message_counts"]["unknown_999"] == 7
    assert "movement_unavailable" in codes(quality, "undetermined")
    assert all(s["kind"] == "unknown" for s in detect_slots(raw)["slots"])


def test_accel_gap_does_not_bridge_missing_or_mixed_axes(tmp_path):
    raw = raw_fixture(tmp_path)
    rows = raw["accelerometer"]
    raw["accelerometer"] = [r for r in rows if not 15 <= r["elapsed_s"] < 25]
    quality = assess_strength_quality(raw)
    assert quality["accelerometer"]["large_gap_count"] == 1
    assert quality["accelerometer"]["supported_duration_s"] == pytest.approx(289.8)
    assert "movement_gaps" in codes(quality, "caution")


def test_overlap_gap_precision_invalid_bounds_no_repair(tmp_path):
    raw = raw_fixture(tmp_path, acceleration=False)
    template = raw["device_sets"][0]
    raw["device_sets"] = [
        {
            **template,
            "device_set_id": f"invented-{i}",
            "start_s": a,
            "end_s": b,
            "duration_s": None if a is None else b - a,
        }
        for i, (a, b) in enumerate(
            [(0, 40.4), (40, 60), (63, 100), (70, 80), (101, 120), (290, 305), (None, None)]
        )
    ]
    original = copy.deepcopy(raw)
    structure = assess_strength_quality(raw)["device_structure"]
    assert len(structure["overlaps"]) == 2  # includes nested interval
    assert [x["delta_s"] for x in structure["gaps"]] == [3, 1]
    assert len(structure["precision_ambiguities"]) == 2
    assert structure["invalid_set_ids"] == ["invented-5", "invented-6"]
    assert raw == original


def test_relative_duration_with_mad_zero_and_min_support(tmp_path):
    raw = raw_fixture(tmp_path, acceleration=False)
    template = raw["device_sets"][0]
    raw["device_sets"] = [
        {**template, "device_set_id": f"invented-{i}", "duration_s": d}
        for i, d in enumerate([20, 20, 20, 20, 20, 200])
    ]
    quality = assess_strength_quality(raw)
    check = quality["duration_checks"][0]
    assert check["median_s"] == 20 and check["mad_s"] == 0
    assert check["upper_limit_s"] == 70 and check["long_set_ids"] == ["invented-5"]
    raw["device_sets"] = raw["device_sets"][:3] + raw["device_sets"][-1:]
    check = assess_strength_quality(raw)["duration_checks"][0]
    assert check["upper_limit_s"] is None and check["long_set_ids"] == []


def test_categories_scalar_array_unknown_preserved_and_no_confirmation():
    rows = normalize_categories([28, 64000, 0, 65534])
    assert [r["original_value"] for r in rows] == [28, 64000, 0, 65534]
    assert [r["native_name"] for r in rows] == ["squat", None, "bench_press", "unknown"]
    assert rows[0]["movement_families"] == ["ginocchio", "anca"]
    assert rows[1]["movement_families"] == rows[3]["movement_families"] == []
    assert normalize_categories("bench_press")[0]["simple_name"] == rows[2]["simple_name"]
    assert normalize_categories(None) == []
    assert normalize_categories("bench_press")[0]["category_code"] == 0
    assert rows[1]["category_code"] == 64000
    assert all(r["provenance"] == "device_hint" and not r["confirmed"] for r in rows)


def test_numeric_array_fit_reaches_matcher_without_changing_raw(tmp_path):
    catalog = StrengthCatalog(tmp_path)
    key = catalog.import_fit(
        strength_fit_bytes(acceleration=False, category_codes=[0, 28, 64000]), synthetic=True
    )
    original = catalog.raw(key)
    quality = catalog.quality(key)
    assert original["device_sets"][0]["category_device"] == [0, 28, 64000]
    assert quality["native_categories"][0]["candidates"][0]["native_name"] == "bench_press"
    slot = next(s for s in catalog.latest(key)["slots"] if s["device_set_refs"])
    result = catalog.suggestions(key, slot["slot_id"])
    assert {r["exercise_id"] for r in result["top3"]} >= {"bench_press", "squat"}
    assert catalog.raw(key) == original
    assert not any(s["confirmed"] for s in catalog.latest(key)["slots"])


def test_quality_export_zero_corrections_append_only_and_old_import(tmp_path):
    catalog = StrengthCatalog(tmp_path)
    key = catalog.import_fit(strength_fit_bytes(acceleration=False), synthetic=True)
    directory = catalog.root / key
    original = {
        n: (directory / n).read_bytes() for n in ("source.fit", "raw.json", "hypotheses.json")
    }
    draft = catalog.latest(key)
    first = catalog.quality(key)
    assert json.loads(catalog.export_quality(key))["quality_id"] == first["quality_id"]
    assert catalog.quality(key) == first
    assert len(list((directory / "quality").glob("*.json"))) == 1
    assert catalog.latest(key) == draft
    assert original == {n: (directory / n).read_bytes() for n in original}
    old_raw = catalog.raw(key)
    del old_raw["message_counts"]
    assert "inventory_unavailable" in codes(assess_strength_quality(old_raw), "undetermined")


def test_zip_fit_same_identity_provenance_and_raw_immutable(tmp_path):
    catalog = StrengthCatalog(tmp_path)
    content = strength_fit_bytes(acceleration=False)
    archive = zip_fixture(content)
    key = catalog.import_fit(archive, synthetic=True)
    original = catalog.raw(key)
    initial = catalog.latest(key)
    assert key == hashlib.sha256(content).hexdigest()
    assert catalog.import_fit(content, synthetic=True) == key
    assert catalog.raw(key) == original and catalog.latest(key) == initial
    assert len(list((catalog.root / key / "inputs").glob("*.json"))) == 2
    assert (catalog.root / key / "source.fit").read_bytes() == content


@pytest.mark.parametrize("member", ["../escape.fit", "/absolute.fit", "C:/escape.fit", "other.txt"])
def test_zip_paths_and_nonfit_rejected(member):
    with pytest.raises(DataError):
        unpack_strength_input(zip_fixture(b"invented", member))


def test_zip_multiple_corrupt_and_decompression_bound():
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("a.fit", b"one")
        archive.writestr("b.fit", b"two")
    with pytest.raises(DataError):
        unpack_strength_input(stream.getvalue())
    # ZIP_STORED allows a deterministic payload corruption, no decompressor dependency.
    content = b"invented FIT payload"
    corrupt = bytearray(zip_fixture(content, compression=0))
    corrupt[corrupt.index(content)] ^= 1
    with pytest.raises(DataError, match="CRC"):
        unpack_strength_input(bytes(corrupt))
    with pytest.raises(DataError, match="decompresso"):
        unpack_strength_input(zip_fixture(b"x" * (MAX_INPUT_BYTES + 1)))


def test_zip_cli_crc_fail_before_output_and_report_hashes(tmp_path):
    source = tmp_path / "s.zip"
    content = strength_fit_bytes(acceleration=False, category_codes=[0, 64000])
    archive = zip_fixture(content)
    source.write_bytes(archive)
    out = tmp_path / "report"
    strength_output(source, out)
    assert source.read_bytes() == archive
    assert (
        json.loads((out / "quality.json").read_text())["source_sha256"]
        == hashlib.sha256(content).hexdigest()
    )
    manifest = json.loads((out / "manifest.json").read_text())
    for name, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    broken = bytearray(content)
    broken[-1] ^= 1
    source.write_bytes(zip_fixture(broken))
    with pytest.raises(DataError, match="CRC"):
        strength_output(source, tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_quality_html_escapes_untrusted_labels(tmp_path):
    raw = raw_fixture(tmp_path, acceleration=False)
    raw["device_sets"][0]["device_set_id"] = '<img src=x onerror="alert(1)">'
    out = tmp_path / "report"
    write_strength_output(raw, detect_slots(raw), out)
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "<img src=x" not in html and "&lt;img src=x" in html


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_hr_gap_s": 0},
        {"max_accel_gap_s": float("nan")},
        {"min_duration_observations": True},
        {"duration_median_ratio": 1},
    ],
)
def test_quality_invalid_config(kwargs):
    with pytest.raises(DataError):
        QualityConfig(**kwargs)


def test_noisy_binary_fixture_quality_and_no_motion_reconstruction(tmp_path):
    raw = raw_fixture(tmp_path, acceleration=False, category_codes=[0, 28, 64000], noisy=True)
    before = copy.deepcopy(raw)
    quality = assess_strength_quality(raw)
    assert quality["hr"]["large_gap_count"] == 1
    assert quality["hr"]["null_values"] == 1
    assert len(quality["device_structure"]["precision_ambiguities"]) == 1
    assert quality["device_structure"]["invalid_set_ids"] == ["fit-set-6"]
    assert quality["duration_checks"][0]["long_set_ids"] == ["fit-set-5"]
    assert quality["accelerometer"]["usable_in_bounds"] == 0
    assert detect_slots(raw)["detection_mode"] == "device_hints"
    assert raw == before


def test_zip_encrypted_and_symlink_rejected():
    from zipfile import ZipInfo

    info = ZipInfo("link.fit")
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr(info, b"invented")
    with pytest.raises(DataError):
        unpack_strength_input(stream.getvalue())
    data = bytearray(zip_fixture(b"invented", compression=0))
    # Mark both headers as encrypted without creating a real cipher fixture.
    data[6] |= 1
    data[data.index(b"PK\x01\x02") + 8] |= 1
    with pytest.raises(DataError):
        unpack_strength_input(bytes(data))


@pytest.mark.parametrize(
    "missing_start,missing_duration", [(True, False), (False, True), (True, True)]
)
def test_recorded_time_and_inferred_bounds_provenance(tmp_path, missing_start, missing_duration):
    raw = raw_fixture(tmp_path, missing_start=missing_start, missing_duration=missing_duration)
    assert raw["start_utc"] == "2025-02-01T12:00:00+00:00"
    assert raw["duration_s"] == 300
    quality = assess_strength_quality(raw)
    assert quality["time_provenance"] == {
        "start_utc": "inferred" if missing_start else "recorded",
        "duration_s": "inferred" if missing_duration else "recorded",
    }
    assert quality["hr"]["supported_duration_s"] == 300
    assert "session_time" in codes(quality, "caution")
