import copy
import csv
import hashlib
import io
import json
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread

import pytest

from hybrid_training_analytics.cli import main
from hybrid_training_analytics.local_web import LocalServer
from hybrid_training_analytics.models import DataError
from hybrid_training_analytics.reconciliation import apply_edit, detect_slots, validate_context
from hybrid_training_analytics.strength import read_strength
from hybrid_training_analytics.strength_catalog import StrengthCatalog
from hybrid_training_analytics.strength_fit import read_strength_fit
from hybrid_training_analytics.strength_synthetic import strength_fit_bytes


def edit(catalog, key, operation):
    return catalog.edit(key, catalog.latest(key)["revision_id"], operation)


def reviewed(catalog, key, body_mass=None):
    original = catalog.latest(key)
    for slot in original["slots"]:
        if slot["kind"] == "work_candidate":
            edit(
                catalog,
                key,
                {"type": "assign", "slot_id": slot["slot_id"], "exercise_id": "bench_press"},
            )
            edit(
                catalog,
                key,
                {
                    "type": "confirm",
                    "slot_id": slot["slot_id"],
                    "values": {"reps": 8, "external_load_kg": 45, "body_mass_kg": body_mass},
                },
            )
        else:
            edit(catalog, key, {"type": "classify", "slot_id": slot["slot_id"], "kind": "rest"})
    return edit(catalog, key, {"type": "finalize"})


def test_catalog_append_only_dedup_concurrency_and_export(tmp_path):
    catalog = StrengthCatalog(tmp_path)
    content = strength_fit_bytes()
    key = catalog.import_fit(content, synthetic=True)
    directory = catalog.root / key
    untouched = {
        name: (directory / name).read_bytes()
        for name in ("source.fit", "raw.json", "hypotheses.json")
    }
    initial = catalog.latest(key)
    assert catalog.import_fit(content, synthetic=True) == key
    assert catalog.latest(key) == initial
    with pytest.raises(DataError):
        catalog.export(key)
    added = edit(catalog, key, {"type": "add"})
    count = len(list((directory / "revisions").glob("*.json")))
    with pytest.raises(DataError, match="Revisione cambiata"):
        catalog.edit(key, initial["revision_id"], {"type": "add"})
    assert len(list((directory / "revisions").glob("*.json"))) == count
    assert added["parent_revision_id"] == initial["revision_id"]
    # A missing untimed set may be explicitly excluded; its time remains unknown.
    missing = next(s for s in added["slots"] if s["start_s"] is None)
    edit(catalog, key, {"type": "classify", "slot_id": missing["slot_id"], "kind": "excluded"})
    result = reviewed(catalog, key)
    payload = json.loads(catalog.export(key))
    assert result["review_status"] == "reviewed"
    work = [s for s in payload["slots"] if s["kind"] == "work"]
    assert len(work) == 2 and all(s["reps"] == 8 for s in work)
    assert work[0]["device_reps"] == 10 and work[0]["external_load_kg"] == 45
    assert work[1]["device_reps"] is None and work[1]["rir"] is None
    with pytest.raises(DataError, match="CSV v1"):
        catalog.export(key, "csv")
    for s in catalog.latest(key)["slots"]:
        if s["kind"] == "work":
            edit(
                catalog,
                key,
                {
                    "type": "confirm",
                    "slot_id": s["slot_id"],
                    "values": {"reps": 8, "external_load_kg": 45, "body_mass_kg": 80},
                },
            )
    edit(catalog, key, {"type": "finalize"})
    # Later context changes cannot reinterpret an already confirmed load mode.
    context = catalog.context()
    context["exercises"][0]["load_mode"] = "bodyweight"
    catalog.save_context(context)
    exported = catalog.export(key, "csv")
    rows = list(csv.DictReader(io.StringIO(exported.decode())))
    assert len(rows) == 2 and rows[0]["load_mode"] == "external"
    assert rows[0]["timestamp"] == "2025-02-01T12:00:30+00:00"
    path = tmp_path / "strength.csv"
    path.write_bytes(exported)
    assert len(read_strength(path)) == 2
    assert untouched == {name: (directory / name).read_bytes() for name in untouched}
    assert hashlib.sha256((directory / "source.fit").read_bytes()).hexdigest() == key


def test_cli_strength_demo_crc_failure_and_existing_output(tmp_path):
    output = tmp_path / "demo"
    assert main(["strength-reconcile-demo", "--out", str(output)]) == 0
    original = (output / "demo.json").read_bytes()
    assert main(["strength-reconcile-demo", "--out", str(output)]) == 2
    assert (output / "demo.json").read_bytes() == original
    description = json.loads(original)
    catalog = StrengthCatalog(Path(description["workspace"]))
    assert catalog.raw(description["session_id"])["synthetic"] is True
    source = output / "strength_synthetic.fit"
    parsed = tmp_path / "parsed"
    assert main(["strength-fit", str(source), "--out", str(parsed)]) == 0
    assert json.loads((parsed / "reconciliation.json").read_text())["review_status"] == "draft"
    broken = bytearray(source.read_bytes())
    broken[-1] ^= 1
    source = tmp_path / "broken.fit"
    source.write_bytes(broken)
    destination = tmp_path / "no-report"
    assert main(["strength-fit", str(source), "--out", str(destination)]) == 2
    assert not destination.exists()


def test_motion_gaps_remain_unknown_and_hr_not_classifier(tmp_path):
    path = tmp_path / "s.fit"
    path.write_bytes(strength_fit_bytes())
    raw = read_strength_fit(path, synthetic=True)
    raw["accelerometer"] = [r for r in raw["accelerometer"] if not 40 <= r["elapsed_s"] < 50]
    raw["records"] = [{**r, "heart_rate_bpm": None} for r in raw["records"]]
    result = detect_slots(raw)
    assert any(
        s["kind"] == "unknown" and s["start_s"] == 40 and s["end_s"] == 50 for s in result["slots"]
    )
    assert all(b["hr_mean_bpm"] is None for b in result["features"])
    assert sum(s["end_s"] - s["start_s"] for s in result["slots"]) == 300


def test_reassignment_clears_declarations_and_untimed_merge_rejected(tmp_path):
    catalog = StrengthCatalog(tmp_path)
    key = catalog.import_fit(strength_fit_bytes(), synthetic=True)
    original = reviewed(catalog, key, 80)
    work = next(s for s in original["slots"] if s["kind"] == "work")
    updated = apply_edit(
        original,
        {"type": "assign", "slot_id": work["slot_id"], "exercise_id": "plank"},
        catalog.context(),
    )
    row = next(s for s in updated["slots"] if s["slot_id"] == work["slot_id"])
    assert row["external_load_kg"] is None and row["reps"] is None and not row["confirmed"]
    assert row["exercise_snapshot"]["load_mode"] == "bodyweight"
    unplaced = apply_edit(updated, {"type": "add"}, catalog.context())
    manual = next(s for s in unplaced["slots"] if s["start_s"] is None)
    with pytest.raises(DataError, match="tempi"):
        apply_edit(
            unplaced,
            {"type": "merge", "slot_ids": [work["slot_id"], manual["slot_id"]]},
            catalog.context(),
        )
    bad_context = copy.deepcopy(catalog.context())
    bad_context["exercises"][0]["id"] = "=SUM(1)"
    with pytest.raises(DataError):
        validate_context(bad_context)


def test_strength_routes_auth_whitelist_and_revisions(tmp_path):
    server = LocalServer(tmp_path, 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(method, path, body=None, auth=True, origin=None, content_type=None):
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        headers = {"X-HTA-Token": server.token} if auth else {}
        if origin:
            headers["Origin"] = origin
        if isinstance(body, dict):
            body = json.dumps(body)
            headers["Content-Type"] = "application/json"
        if content_type:
            headers["Content-Type"] = content_type
        connection.request(method, path, body, headers)
        response = connection.getresponse()
        result = response.status, response.read()
        connection.close()
        return result

    try:
        assert request("GET", "/strength", auth=False)[0] == 200
        assert request("GET", "/strength.js", auth=False)[0] == 200
        assert request("GET", "/api/strength-fit/context", auth=False)[0] == 403
        assert (
            request("GET", "/api/strength-fit/sessions", origin="https://external.invalid")[0]
            == 403
        )
        status, body = request("POST", "/api/strength-fit/import", strength_fit_bytes())
        assert status == 200
        key = json.loads(body)["session_id"]
        status, body = request("GET", "/api/strength-fit/draft/" + key)
        draft = json.loads(body)["draft"]
        assert status == 200 and not any(s["confirmed"] for s in draft["slots"])
        quality = json.loads(body)["quality"]
        assert quality["artifact_type"] == "strength_data_quality"
        status, body = request("POST", "/api/strength-fit/export-quality", {"session_id": key})
        assert status == 200 and json.loads(body)["source_sha256"] == key
        assert (
            request("POST", "/api/strength-fit/export-quality", {"session_id": key}, auth=False)[0]
            == 403
        )
        assert (
            request("POST", "/api/strength-fit/export-quality", {"session_id": "../uv.lock"})[0]
            == 400
        )
        assert server.strength_catalog.latest(key) == draft
        target = next(s for s in draft["slots"] if s["kind"] == "work_candidate")
        suggestion_request = {"session_id": key, "slot_id": target["slot_id"]}
        assert (
            request("POST", "/api/strength-fit/suggest", suggestion_request, auth=False)[0] == 403
        )
        status, body = request("POST", "/api/strength-fit/suggest", suggestion_request)
        assert status == 200 and json.loads(body)["abstained"]
        assert request("POST", "/api/strength-fit/context", {"active_plan": ["squat"]})[0] == 200
        status, body = request("POST", "/api/strength-fit/suggest", suggestion_request)
        result = json.loads(body)
        assert status == 200 and result["status"] == "context_only"
        assert result["top3"][0]["exercise_id"] == "squat"
        assert result["revision_id"] == draft["revision_id"]
        assert len(result["context_sha256"]) == 64 and result["history_refs"] == []
        assert server.strength_catalog.latest(key) == draft
        assert request("GET", "/api/strength-sessions", auth=False)[0] == 403
        status, body = request("GET", "/api/strength-sessions/view/fit-" + key)
        assert status == 200 and json.loads(body)["selected_sets"] == []
        assert request("GET", "/api/strength-sessions/view/../uv.lock")[0] == 400
        status, body = request(
            "POST",
            "/api/strength-sessions/save",
            {"session_id": "fit-" + key, "parent_revision_id": None, "values": {}},
        )
        minimal = json.loads(body)
        assert status == 200 and minimal["declarations"]["session_rpe"] is None
        assert (
            request(
                "POST",
                "/api/strength-sessions/save",
                {"session_id": "fit-" + key, "parent_revision_id": None, "values": {}},
            )[0]
            == 400
        )
        assert (
            request("POST", "/api/strength-sessions/export", {"session_id": "fit-" + key})[0] == 200
        )
        assert request("GET", "/api/strength-sessions/ux")[0] == 200
        assert request("GET", "/api/strength-evaluation", auth=False)[0] == 403
        capture = {"session_id": key, "slot_id": target["slot_id"]}
        assert request("POST", "/api/strength-evaluation/capture", capture, auth=False)[0] == 403
        status, body = request("POST", "/api/strength-evaluation/capture", capture)
        trial = json.loads(body)["trial"]
        assert status == 200 and trial["predicted_ids"] == ["squat"]
        status, body = request(
            "POST",
            "/api/strength-evaluation/label",
            {
                "trial_id": trial["trial_id"],
                "parent_revision_id": None,
                "values": {"exercise_id": "squat"},
            },
        )
        assert status == 200
        assert (
            request(
                "POST",
                "/api/strength-evaluation/label",
                {
                    "trial_id": "../uv.lock",
                    "values": {},
                    "parent_revision_id": None,
                },
            )[0]
            == 400
        )
        status, body = request("POST", "/api/strength-evaluation/export", {})
        assert status == 200 and json.loads(body)["top3_hits"] == 1

        assert server.strength_catalog.latest(key) == draft
        mutation = {
            "session_id": key,
            "parent_revision_id": draft["revision_id"],
            "operation": {"type": "add"},
        }
        assert request("POST", "/api/strength-fit/edit", mutation)[0] == 200
        assert request("POST", "/api/strength-fit/edit", mutation)[0] == 400
        assert request("POST", "/api/strength-fit/export", {"session_id": key})[0] == 400
        assert request("GET", "/api/strength-fit/draft/../../uv.lock")[0] == 400
        assert request("GET", "/source.fit")[0] == 404
        assert (
            request("POST", "/api/strength-fit/context", "{}", content_type="text/plain")[0] == 400
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
