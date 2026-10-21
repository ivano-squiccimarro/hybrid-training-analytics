import csv
import hashlib
import json

from hybrid_training_analytics.cli import main


def calibration_arguments(source, output, distance="10800"):
    return [
        "calibrate",
        str(source),
        "--distance-m",
        distance,
        "--distance-source",
        "watch_final",
        "--incline-pct",
        "0",
        "--out",
        str(output),
    ]


def test_calibration_cli_separate_derivative_reproduction_and_no_overwrite(run, tmp_path):
    source = tmp_path / "run.json"
    source.write_text(json.dumps(run.to_dict()), encoding="utf-8")
    original = source.read_bytes()
    output = tmp_path / "scaled"
    assert main(calibration_arguments(source, output)) == 0
    payload = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_run_json_sha256"] == hashlib.sha256(original).hexdigest()
    assert payload["samples"][-1]["scaled_distance_m"] == 10800
    with (output / "records_scaled.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(run.samples)
    assert rows[100]["heart_rate_bpm"] == ""
    assert rows[0]["timestamp_utc"].endswith("+00:00")
    assert not (output / "run.json").exists()
    assert not (output / "INCOMPLETE.txt").exists()
    saved = {path.name: path.read_bytes() for path in output.iterdir()}
    assert main(calibration_arguments(source, output)) == 2
    assert saved == {path.name: path.read_bytes() for path in output.iterdir()}
    replica = tmp_path / "replica"
    assert main(calibration_arguments(source, replica)) == 0
    assert saved == {path.name: path.read_bytes() for path in replica.iterdir()}
    assert source.read_bytes() == original


def test_invalid_calibration_creates_no_output(run, tmp_path):
    source = tmp_path / "run.json"
    source.write_text(json.dumps(run.to_dict()), encoding="utf-8")
    output = tmp_path / "invalid"
    assert main(calibration_arguments(source, output, distance="nan")) == 2
    assert not output.exists()


def test_failed_write_retains_incomplete_marker(run, tmp_path, monkeypatch):
    source = tmp_path / "run.json"
    source.write_text(json.dumps(run.to_dict()), encoding="utf-8")
    output = tmp_path / "interrupted"

    def fail_write(*args, **kwargs):
        raise OSError("Invented disk failure")

    monkeypatch.setattr("hybrid_training_analytics.calibration_io.write_json", fail_write)
    assert main(calibration_arguments(source, output)) == 2
    assert (output / "INCOMPLETE.txt").exists()
    assert not (output / "calibration.json").exists()
