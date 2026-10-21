import json

from hybrid_training_analytics.cli import main


def test_demo_end_to_end_and_reproducible_report(tmp_path):
    output = tmp_path / "demo"
    assert main(["demo", "--out", str(output)]) == 0
    assert "DATI SINTETICI" in (output / "report.html").read_text(encoding="utf-8")
    assert (output / "strength" / "report.html").exists()
    assert not (output / "INCOMPLETE.txt").exists()
    reproduced = tmp_path / "reproduced"
    assert (
        main(
            [
                "report",
                str(output / "run.json"),
                "--config",
                str(output / "analysis.json"),
                "--out",
                str(reproduced),
            ]
        )
        == 0
    )
    assert json.loads((output / "analysis.json").read_text(encoding="utf-8")) == json.loads(
        (reproduced / "analysis.json").read_text(encoding="utf-8")
    )
    assert (output / "report.html").read_bytes() == (reproduced / "report.html").read_bytes()


def test_import_one_command_and_no_overwrite(synthetic_fit, tmp_path):
    output = tmp_path / "imported"
    assert main(["import", str(synthetic_fit), "--out", str(output)]) == 0
    original = (output / "report.html").read_bytes()
    assert main(["import", str(synthetic_fit), "--out", str(output)]) == 2
    assert (output / "report.html").read_bytes() == original


def test_invalid_input_produces_no_output(tmp_path):
    source = tmp_path / "bad.fit"
    source.write_bytes(b"invalid")
    output = tmp_path / "output"
    assert main(["import", str(source), "--out", str(output)]) == 2
    assert not output.exists()


def test_normalized_json_contains_no_coordinates_or_source_path(run):
    payload = json.dumps(run.to_dict())
    assert "latitude" not in payload and "serial_number" not in payload
    assert "C:" not in payload
