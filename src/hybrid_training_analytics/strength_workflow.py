"""Strength CLI and a self-contained synthetic reconciliation workspace."""

from __future__ import annotations

import hashlib
import json
import platform
from html import escape
from pathlib import Path

import fitdecode

from . import __version__
from .reconciliation import detect_slots
from .reporting import new_output, write_json
from .strength_catalog import StrengthCatalog
from .strength_input import read_strength_source
from .strength_quality import assess_strength_quality
from .strength_synthetic import strength_fit_bytes


def write_strength_output(
    raw: dict, draft: dict, output: Path, input_descriptor: dict | None = None
) -> Path:
    new_output(output)
    marker = output / "INCOMPLETE.txt"
    marker.write_text("Generazione forza in corso", encoding="utf-8")
    write_json(output / "raw.json", raw)
    write_json(output / "reconciliation.json", draft)
    quality = assess_strength_quality(raw)
    write_json(output / "quality.json", quality)
    if input_descriptor is not None:
        write_json(output / "input_provenance.json", input_descriptor)
    sections = []
    for key, label in (
        ("available", "Disponibile"),
        ("caution", "Richiede cautela"),
        ("undetermined", "Non determinabile"),
    ):
        sections.append(
            "<section><h2>"
            + label
            + "</h2><ul>"
            + "".join("<li>" + escape(f["text"]) + "</li>" for f in quality["sections"][key])
            + "</ul></section>"
        )
    native_rows = []
    for row in quality["native_categories"]:
        text = "; ".join(
            c["simple_name"]
            + " ["
            + str(c["original_value"])
            + "]"
            + (" · " + "/".join(c["movement_families"]) if c["movement_families"] else "")
            for c in row["candidates"]
        )
        native_rows.append(
            "<tr><td>"
            + escape(row["device_set_id"])
            + "</td><td>"
            + escape(text or "Categoria assente")
            + "</td></tr>"
        )
    rows = []
    for row in draft["slots"]:
        rows.append(
            "<tr>"
            + "".join(
                "<td>" + escape("Non disponibile" if row[k] is None else str(row[k])) + "</td>"
                for k in (
                    "start_s",
                    "end_s",
                    "kind",
                    "device_reps",
                    "device_weight_kg",
                    "confirmed",
                )
            )
            + "</tr>"
        )
    html = (
        '<!doctype html><html lang="it"><meta charset="utf-8">'
        "<title>Forza FIT: qualità dei dati</title>"
        "<style>body{font:16px/1.6 system-ui;max-width:1000px;margin:40px auto;"
        "padding:20px;color:#20343f}td,th{padding:10px;border:1px solid #cbd8dd}"
        "table{border-collapse:collapse;width:100%}</style>"
        "<h1>Forza FIT: qualità dei dati</h1><p>"
        + ("DATI INTERAMENTE SINTETICI · " if raw["synthetic"] else "Dati locali · ")
        + escape(raw["start_utc"])
        + " · durata "
        + str(raw["duration_s"])
        + " s</p>"
        "<p>Modalità: " + escape(draft["detection_mode"]) + "</p>"
        "<p>Nessun blocco o esercizio è confermato automaticamente. FC registrata "
        "in bpm; accelerometro in g quando calibrato, altrimenti conteggi. "
        "Ripetizioni e kg Garmin sono indicazioni non verificate.</p>"
        + "".join(sections)
        + "<h2>Nomi Garmin in parole semplici</h2><p>Candidati del dispositivo, "
        "non esercizi confermati. Famiglie: convenzione del prodotto, non muscoli misurati.</p>"
        "<table><thead><tr><th>Marcatura</th><th>Candidati · codice / famiglia</th></tr></thead>"
        "<tbody>" + "".join(native_rows) + "</tbody></table>"
        "<details><summary>Timeline dettagliata facoltativa</summary>"
        "<table><thead><tr><th>Inizio · s</th><th>Fine · s</th><th>Ipotesi</th>"
        "<th>Ripetizioni Garmin</th><th>Carico Garmin · kg</th><th>Confermato</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></details><h2>Limiti</h2><ul>"
        + "".join("<li>" + escape(x) + "</li>" for x in raw["warnings"] + draft["limits"])
        + "</ul><details><summary>Qualità, parametri e provenienza</summary><pre>"
        + escape(json.dumps(quality, ensure_ascii=False, indent=2))
        + "</pre></details><p>La FC non identifica serie o esercizi. Nessuna probabilità "
        "calibrata, diagnosi o prescrizione. Revisione guidata: app locale /strength.</p></html>"
    )
    (output / "report.html").write_text(html, encoding="utf-8")
    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "strength_reconciliation_manifest",
            "source_sha256": raw["source_sha256"],
            "synthetic": raw["synthetic"],
            "detection_config": draft["detection_config"],
            "quality_config": quality["config"],
            "versions": {
                "hta": __version__,
                "python": platform.python_version(),
                "fitdecode": fitdecode.__version__,
            },
            "code_sha256": {
                name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                for name in (
                    "strength_fit.py",
                    "reconciliation.py",
                    "strength_quality.py",
                    "exercise_vocabulary.py",
                    "strength_input.py",
                    "strength_workflow.py",
                )
            },
            "output_sha256": {
                name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in (
                    "raw.json",
                    "reconciliation.json",
                    "quality.json",
                    "report.html",
                    *(["input_provenance.json"] if input_descriptor is not None else []),
                )
            },
        },
    )
    marker.unlink()
    return (output / "report.html").resolve()


def strength_output(source: Path, output: Path) -> Path:
    # Decode/CRC errors fail before creating the destination.
    raw, descriptor = read_strength_source(source)
    return write_strength_output(raw, detect_slots(raw), output, descriptor)


def strength_demo(output: Path) -> Path:
    # Validate synthetic data before opening a fresh directory; never overwrite.
    content = strength_fit_bytes()
    new_output(output)
    marker = output / "INCOMPLETE.txt"
    marker.write_text("Demo forza sintetica in corso", encoding="utf-8")
    (output / "strength_synthetic.fit").write_bytes(content)
    catalog = StrengthCatalog(output / "workspace")
    key = catalog.import_fit(content, synthetic=True)
    noisy = strength_fit_bytes(acceleration=False, category_codes=[0, 28, 64000], noisy=True)
    (output / "strength_noisy_synthetic.fit").write_bytes(noisy)
    noisy_key = catalog.import_fit(noisy, synthetic=True)
    write_strength_output(
        catalog.raw(noisy_key), catalog.latest(noisy_key), output / "inspection-noisy"
    )
    context = catalog.context()
    context["active_plan"] = ["bench_press", "leg_press", "lat_pulldown"]
    context["history"] = [
        {
            "exercise_id": "bench_press",
            "timestamp_utc": f"2025-01-{day}T12:00:00+00:00",
            "reps": 10,
            "load_kg": 50,
            "duration_s": 30,
            "previous_exercise_id": None,
        }
        for day in (15, 20, 25)
    ]
    catalog.save_context(context)
    report = write_strength_output(catalog.raw(key), catalog.latest(key), output / "inspection")
    from .strength_sessions import StrengthSessions

    minimal = StrengthSessions(catalog.workspace)
    minimal.save("fit-" + key, None, {})
    manual = minimal.save(None, None, {}, synthetic=True)
    write_json(
        output / "demo.json",
        {
            "synthetic": True,
            "session_id": key,
            "quality_noisy_session_id": noisy_key,
            "minimal_fit_session_id": "fit-" + key,
            "minimal_manual_session_id": manual["session_id"],
            "workspace": str(catalog.workspace),
            "known_synthetic_work_intervals_s": [[30, 60], [180, 210]],
            "omitted_device_set": "second work interval deliberately omitted",
            "local_ui_command": "uv run --locked python -m hybrid_training_analytics serve "
            '--workspace "' + str(catalog.workspace) + '" --port 0',
            "ui_route": "/strength",
            "report": str(report),
        },
    )
    marker.unlink()
    return report
