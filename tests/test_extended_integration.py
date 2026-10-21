import hashlib
import http.client
import json
from threading import Thread

import pytest

from hybrid_training_analytics.advanced_metrics import DecouplingConfig
from hybrid_training_analytics.cli import main
from hybrid_training_analytics.extended_reporting import enrich, write_extended
from hybrid_training_analytics.local_catalog import LocalCatalog
from hybrid_training_analytics.local_web import LocalServer
from hybrid_training_analytics.models import DataError


def test_extended_reproduction_and_preserve(run, tmp_path):
    before = run.to_dict()
    annotation = {
        "sensor": "wrist",
        "notes": "<script>alert(1)</script>",
        "target_distance_m": 6000,
        "distance_source": "user_other",
    }
    payload = enrich(run, annotation, DecouplingConfig())
    for name in ("a", "b"):
        write_extended(run, payload, tmp_path / name, input_sha256="f" * 64)
    for filename in ("analysis_extended.json", "report.html", "minutes.csv", "manifest.json"):
        assert (tmp_path / "a" / filename).read_bytes() == (tmp_path / "b" / filename).read_bytes()
    assert run.to_dict() == before
    report = (tmp_path / "a" / "report.html").read_text(encoding="utf-8")
    assert "<script>alert" not in report
    assert "&lt;script&gt;" in report
    with pytest.raises(DataError):
        write_extended(run, payload, tmp_path / "a", input_sha256="f" * 64)


def test_catalog_dedup_revision_analysis_and_traversal(synthetic_fit, tmp_path):
    catalog = LocalCatalog(tmp_path)
    content = synthetic_fit.read_bytes()
    key = catalog.import_fit(content, synthetic=True)
    assert catalog.import_fit(content, synthetic=True) == key
    catalog.annotate("run:" + key, {"sensor": "wrist"})
    catalog.annotate("run:" + key, {"sensor": "chest"})
    assert len(list(catalog.annotation_directory("run:" + key).glob("*.json"))) == 2
    assert catalog.runs()[0]["annotations"]["sensor"] == "chest"
    first = catalog.analyze(key, {"start_s": 300})
    second = catalog.analyze(key, {"start_s": 300})
    assert first["output_id"] != second["output_id"]
    path = catalog.output_file(first["output_id"], "minutes.csv")
    assert path.exists()
    assert catalog.latest_analysis(key)["source_sha256"] == key
    assert (catalog.root / "runs" / key / "source.fit").read_bytes() == content
    for bad in ("../outside", "full-" + key + "-../" + "a" * 32):
        with pytest.raises(DataError):
            catalog.output_file(bad, "minutes.csv")
    with pytest.raises(DataError):
        catalog.output_file(first["output_id"], "source.fit")


def test_corrupt_fit_does_not_enter_catalog(synthetic_fit, tmp_path):
    catalog = LocalCatalog(tmp_path)
    content = bytearray(synthetic_fit.read_bytes())
    content[-1] ^= 1
    with pytest.raises(DataError):
        catalog.import_fit(bytes(content))
    assert catalog.runs() == []


def test_http_local_auth_origin_and_whitelist(synthetic_fit, tmp_path):
    server = LocalServer(tmp_path, 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        value = response.read()
        status = response.status
        security = dict(response.getheaders())
        connection.close()
        return status, value, security

    try:
        assert server.server_address[0] == "127.0.0.1"
        status, body, headers = request("GET", "/")
        assert status == 200 and server.token.encode() not in body
        assert "frame-ancestors" in headers["Content-Security-Policy"]
        assert request("GET", "/api/runs")[0] == 403
        auth = {"X-HTA-Token": server.token}
        assert (
            request("GET", "/api/runs", headers={**auth, "Origin": "https://external.invalid"})[0]
            == 403
        )
        assert request("GET", "/api/runs", headers={**auth, "Host": "external.invalid"})[0] == 403
        assert request("GET", "/api/runs", headers=auth)[0] == 200
        assert request("GET", "/../uv.lock", headers=auth)[0] == 404
        status, body, _ = request("POST", "/api/import-fit", synthetic_fit.read_bytes(), auth)
        assert status == 200
        key = json.loads(body)["run_id"]
        assert key == hashlib.sha256(synthetic_fit.read_bytes()).hexdigest()
        headers = {**auth, "Content-Type": "application/json"}
        assert (
            request(
                "POST",
                "/api/analyze",
                json.dumps({"run_id": key, "config": {"start_s": 300}}),
                headers,
            )[0]
            == 200
        )
        assert (
            request(
                "POST",
                "/api/annotate",
                json.dumps({"subject": "run:" + key, "values": {"sensor": "wrist"}}),
                headers,
            )[0]
            == 200
        )
        assert (
            request(
                "POST",
                "/api/download",
                json.dumps({"output_id": "../../uv.lock", "filename": "manifest.json"}),
                headers,
            )[0]
            == 400
        )
        assert (
            request("POST", "/api/analyze", "{}", {**headers, "Content-Length": "10485761"})[0]
            == 400
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_full_demo_and_all_cli_flows(tmp_path):
    demo = tmp_path / "demo"
    assert main(["demo-full", "--out", str(demo)]) == 0
    original = (demo / "demo.json").read_bytes()
    payload = json.loads(original)
    assert payload["synthetic"] is True
    assert payload["hybrid"]["strength_trends"][0]["observations"][1]["delta_load_kg"] == 5
    assert payload["comparison"]["status"] == "available"
    assert payload["comparison"]["delta_cadence_pct"] == 0
    assert main(["demo-full", "--out", str(demo)]) == 2
    assert (demo / "demo.json").read_bytes() == original
    catalog = LocalCatalog(demo / "workspace")
    key = catalog.runs()[0]["run_id"]
    source = catalog.root / "runs" / key
    signals = tmp_path / "signals"
    assert main(["signals", str(source / "source.fit"), "--out", str(signals)]) == 0
    annotated = tmp_path / "annotations.json"
    annotated.write_text(json.dumps({"sensor": "wrist"}))
    enriched = tmp_path / "enriched"
    assert (
        main(
            [
                "enrich",
                str(source / "run.json"),
                "--signals",
                str(source / "signals.json"),
                "--annotations",
                str(annotated),
                "--out",
                str(enriched),
            ]
        )
        == 0
    )
    a = catalog.reports / payload["analyses"][0]["output_id"] / "analysis_extended.json"
    b = catalog.reports / payload["analyses"][1]["output_id"] / "analysis_extended.json"
    assert main(["compare-phases", str(a), str(b), "--out", str(tmp_path / "comparison")]) == 0
    assert (
        main(
            [
                "hybrid-report",
                str(demo / "inputs" / "strength_synthetic.csv"),
                "--runs",
                str(a),
                str(b),
                "--out",
                str(tmp_path / "hybrid"),
            ]
        )
        == 0
    )


def test_strength_duplicate_and_invalid_interval_preserve(tmp_path):
    catalog = LocalCatalog(tmp_path)
    content = (
        b"timestamp,session_id,exercise,load_mode,body_mass_kg,added_load_kg,"
        b"external_load_kg,set_index,reps,rir,rpe\n"
        b"2025-01-01T12:10:00+00:00,s,squat,external,80,0,50,1,5,2,8\n"
    )
    catalog.import_strength(content)
    with pytest.raises(DataError):
        catalog.import_strength(content.replace(b",50,", b",55,"))
    with pytest.raises(DataError):
        catalog.annotate(
            "strength:s",
            {"start_utc": "2025-01-01T13:00:00+00:00", "end_utc": "2025-01-01T14:00:00+00:00"},
        )
    assert len(catalog.strength_sets()) == 1
    assert catalog.annotations("strength:s")["start_utc"] is None


def test_revision_sequence_keeps_latest_with_equal_or_backward_clock(tmp_path, monkeypatch):
    from datetime import UTC
    from datetime import datetime as RealDatetime
    from types import SimpleNamespace

    import hybrid_training_analytics.annotations as annotations
    import hybrid_training_analytics.strength_catalog as strength_catalog
    import hybrid_training_analytics.strength_evaluation as strength_evaluation
    import hybrid_training_analytics.strength_sessions as strength_sessions
    from hybrid_training_analytics.strength_synthetic import strength_fit_bytes

    clock = RealDatetime(2025, 2, 2, tzinfo=UTC)

    class FixedClock:
        @staticmethod
        def now(tz):
            return clock

        fromisoformat = staticmethod(RealDatetime.fromisoformat)

    for module in (annotations, strength_catalog, strength_sessions, strength_evaluation):
        monkeypatch.setattr(module, "datetime", FixedClock)
    # UUID lexical order opposes save order; timestamps alone cannot order revisions.
    ids = iter(["f" * 32, "e" * 32, "d" * 32, "c" * 32])
    monkeypatch.setattr(annotations, "uuid4", lambda: SimpleNamespace(hex=next(ids)))
    directory = tmp_path / "legacy-annotations"
    first = annotations.append_revision(directory, "strength:test", {"sensor": "wrist"})
    second = annotations.append_revision(directory, "strength:test", {"sensor": "chest"})
    assert first["created_utc"] == second["created_utc"]
    assert max([first, second], key=annotations.revision_key) == second
    assert first["revision_sequence"] == 1 and second["revision_sequence"] == 2
    clock = RealDatetime(2025, 2, 1, tzinfo=UTC)
    third = annotations.append_revision(directory, "strength:test", {})
    assert max([first, second, third], key=annotations.revision_key) == third
    # Same ordering protects minimal sessions, contexts, FIT drafts, and trial labels.
    fit = strength_catalog.StrengthCatalog(tmp_path)
    key = fit.import_fit(strength_fit_bytes(), synthetic=True)
    c = fit.save_context({"active_plan": ["squat"]})
    d = fit.save_context({"active_plan": ["bench_press"]})
    assert c["created_utc"] == d["created_utc"]
    assert fit.context()["active_plan"] == ["bench_press"]
    old = fit.latest(key)
    changed = fit.edit(key, old["revision_id"], {"type": "add"})
    assert fit.latest(key)["revision_id"] == changed["revision_id"]
    sessions = strength_sessions.StrengthSessions(tmp_path)
    a = sessions.save(None, None, {}, synthetic=True)
    b = sessions.save(a["session_id"], a["revision_id"], {"sensor": "chest"})
    assert sessions.latest(a["session_id"])["revision_id"] == b["revision_id"]
    engine = strength_evaluation.StrengthEvaluation(tmp_path)
    target = next(s for s in changed["slots"] if s["kind"] == "work_candidate")
    trial = engine.capture(key, target["slot_id"])
    first_label = engine.label(trial["trial_id"], None, {"exercise_id": "squat"})
    m = engine.label(trial["trial_id"], first_label["revision_id"], {})
    assert engine.latest_label(trial["trial_id"])["revision_id"] == m["revision_id"]
