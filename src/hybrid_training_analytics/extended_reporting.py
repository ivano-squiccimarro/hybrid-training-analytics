"""Offline extended report and reproducibility manifest, separate from bootstrap outputs."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from .advanced_metrics import (
    DecouplingConfig,
    channels,
    decoupling,
    minute_bins,
    sensitivity,
    support,
)
from .annotations import validate_annotations
from .calibration import CalibrationConfig, calibrate
from .calibration_io import write_calibration
from .hybrid import session_load
from .models import RunData
from .reporting import new_output, write_json


def enrich(
    run: RunData, annotations: dict, config: DecouplingConfig, signals: dict | None = None
) -> dict:
    values = validate_annotations(annotations)
    calibration = None
    factor = 1.0
    if values["target_distance_m"] is not None:
        calibration = calibrate(
            run,
            CalibrationConfig(
                values["target_distance_m"],
                values["distance_source"],
                values["incline_pct"],
                config.max_gap_s,
            ),
        )
        factor = calibration["scale_factor"]
    bins = minute_bins(run, signals, factor, config.max_gap_s)
    rows = channels(run, signals, factor)
    exposure = support(
        run, rows, ("heart_rate_bpm",), 0, run.samples[-1].elapsed_s, config.max_gap_s
    )
    return {
        "schema_version": 1,
        "artifact_type": "extended_run_analysis",
        "source_sha256": run.source_sha256,
        "synthetic": run.synthetic,
        "start_utc": run.samples[0].timestamp_utc,
        "duration_s": run.samples[-1].elapsed_s,
        "record_distance_delta_m": None
        if run.samples[0].distance_m is None or run.samples[-1].distance_m is None
        else run.samples[-1].distance_m - run.samples[0].distance_m,
        "annotations": values,
        "config": asdict(config),
        "cardiovascular_exposure": {
            "mean_hr_bpm": exposure["means"]["heart_rate_bpm"],
            "valid_s": exposure["valid_s"],
            "coverage_fraction": exposure["coverage_fraction"],
            "hr_integral_bpm_s": exposure["integrals"]["heart_rate_bpm"],
            "provenance": "time_integral_recorded_hr; not zone load or TRIMP",
        },
        "session_load": session_load(values, duration_s=run.samples[-1].elapsed_s),
        "decoupling": decoupling(run, config, signals, factor),
        "sensitivity": sensitivity(run, config, signals, factor),
        "calibration": calibration,
        "minutes": bins,
        "provenance": {
            "heart_rate_bpm": "FIT recorded",
            "speed_m_s": "FIT device estimate" if factor == 1 else "uniform scaling estimate",
            "cadence_fit_rpm": "native FIT rpm, no inferred steps",
            "power_w": "FIT recorded device power",
            "annotations": "user declared",
            "decoupling": "calculated descriptor",
        },
    }


def _chart(minutes: list[dict]) -> str:
    values = [r["heart_rate_bpm"] for r in minutes if r["heart_rate_bpm"] is not None]
    if not values:
        return "<p>FC non disponibile: nessun intervallo valido.</p>"
    lo, hi = min(values) - 5, max(values) + 5
    end = minutes[-1]["end_s"]
    elements = [
        '<svg viewBox="0 0 900 220" role="img" aria-label="FC media temporale per minuto, bpm">',
        '<rect width="900" height="220" fill="#f3f7fa"/>',
        f'<text x="8" y="20">FC media per minuto · bpm ({lo:.0f}–{hi:.0f})</text>',
        '<text x="660" y="210">Tempo trascorso · min</text>',
    ]
    for row in minutes:
        value = row["heart_rate_bpm"]
        x = 50 + 800 * row["start_s"] / end
        width = 800 * row["span_s"] / end
        if value is None:
            elements.append(
                f'<rect x="{x:.2f}" y="30" width="{width:.2f}" height="140" fill="#ddd"/>'
            )
        else:
            y = 170 - 140 * (value - lo) / (hi - lo)
            elements.append(
                f'<line x1="{x:.2f}" x2="{x + width:.2f}" y1="{y:.2f}" y2="{y:.2f}" '
                'stroke="#187d8d" stroke-width="3"/>'
            )
    for minute in range(0, int(end / 60) + 1, 5):
        elements.append(f'<text x="{50 + 800 * minute * 60 / end:.2f}" y="190">{minute}</text>')
    return (
        "".join(elements)
        + "</svg><p>Segmenti separati: medie sui soli secondi validi; grigio = nessun supporto.</p>"
    )


def write_extended(
    run: RunData, payload: dict, output: Path, *, input_sha256: str, signals: dict | None = None
) -> Path:
    new_output(output)
    marker = output / "INCOMPLETE.txt"
    marker.write_text("Generazione estesa in corso.\n", encoding="utf-8")
    write_json(output / "run.json", run.to_dict())
    write_json(output / "analysis_extended.json", payload)
    if signals is not None:
        write_json(output / "signals.json", signals)
    with (output / "minutes.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(payload["minutes"][0]))
        writer.writeheader()
        writer.writerows(payload["minutes"])
    if payload["calibration"] is not None:
        sub = output / "calibration"
        sub.mkdir()
        write_calibration(payload["calibration"], sub, input_sha256)
    write_descriptor_report(payload, output / "report.html", chart=_chart(payload["minutes"]))
    lock = Path(__file__).resolve().parents[2] / "uv.lock"
    manifest = {
        "schema_version": 1,
        "algorithm": "extended_run_v1",
        "software_version": version("hybrid-training-analytics"),
        "domain_code_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in (
                "advanced_metrics.py",
                "annotations.py",
                "signals.py",
                "hybrid.py",
                "calibration.py",
                "extended_reporting.py",
            )
        },
        "source_sha256": run.source_sha256,
        "input_sha256": input_sha256,
        "signals_sha256": None
        if signals is None
        else hashlib.sha256(
            json.dumps(signals, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "versions": {name: version(name) for name in ("fitdecode", "matplotlib", "tzdata")},
        "config": payload["config"],
        "annotations": payload["annotations"],
    }
    manifest["output_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.iterdir()
        if p.is_file() and p != marker
    }
    write_json(output / "manifest.json", manifest)
    marker.unlink()
    return (output / "report.html").resolve()


def write_descriptor_report(payload: dict, path: Path, *, chart: str = "") -> None:
    if payload.get("artifact_type") == "strength_suggestion_evaluation":
        with path.open("x", encoding="utf-8") as out:
            out.write(evaluation_document(payload))
        return
    if payload.get("artifact_type") == "hybrid_descriptive_load":
        with path.open("x", encoding="utf-8") as out:
            out.write(hybrid_document(payload))
        return
    synthetic = " · INTERAMENTE SINTETICO" if payload.get("synthetic") else ""
    coupling = payload.get("decoupling", {})
    value = coupling.get("decoupling_pct")
    label = "Non disponibile" if value is None else f"{value:.2f} %"
    reasons = ", ".join(coupling.get("reasons", []))
    title = html.escape(payload.get("artifact_type", "Analisi descrittiva"))
    document = f"""<!doctype html><html lang="it"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>{title}</title><style>body{{font:16px system-ui;margin:40px auto;
    max-width:1080px;padding:0 20px;color:#183341}}
    h1{{font-size:28px}}svg{{width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f7fa;padding:20px}}
    .note{{border-left:4px solid #187d8d;padding:16px}}</style>
    <h1>Analisi locale{synthetic}</h1><p>{title}</p>
    <p class="note">Dati registrati, stime, dichiarazioni e inferenze sono distinti.
    Null significa sconosciuto. Nessuna prescrizione o causa fisiologica attribuita.</p>
    <h2>Decoupling: {label}</h2><p>{html.escape(reasons)}</p>{chart}
    <h2>Parametri, unità, supporti e risultati</h2>
    <pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))}
    </pre></html>"""
    with path.open("x", encoding="utf-8") as out:
        out.write(document)


def hybrid_document(payload: dict) -> str:
    def value(v):
        return html.escape("Ignoto" if v is None else str(v))

    rows = []
    for s in payload["strength_sessions"]:
        volumes = (
            "; ".join(
                r.get("exercise_name", r["exercise"])
                + ": "
                + str(r["external_volume_kg_reps"])
                + " kg·reps esterni"
                + (
                    "; " + str(r["system_mass_proxy_kg_reps"]) + " kg·reps proxy massa sistema"
                    if r["system_mass_proxy_kg_reps"] is not None
                    else ""
                )
                for r in s["exercises"]
            )
            or "Ignoto: nessuna serie nota"
        )
        row = [
            s["temporal_start_utc"],
            {
                "upper": "Parte superiore",
                "lower": "Parte inferiore",
                "full_body": "Corpo intero",
            }.get(s["body_scope_declared"], s["body_scope_declared"]),
            s["elapsed_duration_s"],
            s["declared_duration_s"],
            s["hr_support_s"],
            s["load"]["session_rpe"],
            s["load"]["session_rpe_load_au"],
            s["known_set_count"],
            volumes,
        ]
        rows.append("<tr>" + "".join("<td>" + value(v) + "</td>" for v in row) + "</tr>")
    headers = [
        "Inizio · UTC",
        "Tipo dichiarato",
        "Durata registrata · s",
        "Durata dichiarata · s",
        "Supporto FC · s",
        "RPE seduta · 1–10",
        "RPE × min · AU",
        "Serie note",
        "Volume delle sole serie note",
    ]
    return (
        '<!doctype html><html lang="it"><meta charset="utf-8"><meta name="viewpor'
        't" content="width=device-width,initial-scale=1">'
        "<title>Diario ibrido locale</title><style>body{font:16px/1.6 system-ui;c"
        "olor:#20343f;max-width:1280px;margin:30px auto;padding:20px}"
        "table{border-collapse:collapse;width:100%}td,th{padding:12px;border:1px "
        "solid #ccd8dd;text-align:left}.table{overflow:auto}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef4f5;padding:20px}</style>"
        "<h1>Diario forza e contesto corsa</h1>"
        + (
            "<p>DATI INTERAMENTE SINTETICI</p>"
            if payload.get("synthetic")
            else "<p>Catalogo locale · dati personali o di origine mista</p>"
        )
        + "<p>Registrazioni, dichiarazioni e inferenze restano distinte. La durata "
        "dell’orologio non è tempo di lavoro muscolare; "
        "il supporto FC non dimostra accuratezza. Nessun giudizio di recupero o causalità.</p>"
        "<p>Sedute forza: "
        + str(payload["strength_session_count"])
        + ". Senza data: "
        + str(len(payload["undated_strength_session_ids"]))
        + "</p>"
        '<div class="table"><table><thead><tr>'
        + "".join("<th>" + value(v) + "</th>" for v in headers)
        + "</tr></thead><tbody>"
        + ("".join(rows) or '<tr><td colspan="9">Nessuna seduta importata o salvata.</td></tr>')
        + "</tbody></table></div>"
        "<p>I volumi riguardano le sole serie note per esercizio; il volume dell’"
        "intera seduta resta ignoto. "
        "Identità FIT/CSV e collegamenti manuali espliciti evitano doppio contegg"
        "io. Set RPE non sostituisce RPE di seduta.</p>"
        "<details><summary>Settimane, confronto temporale, provenienza e limiti</summary><pre>"
        + html.escape(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2))
        + "</pre></details></html>"
    )


def evaluation_document(payload: dict) -> str:
    rows = []
    for title, result in payload.get("cohorts", {"Tutte le prove": payload}).items():
        rows.append(
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(value)) + "</td>"
                for value in (
                    {"synthetic": "Prove sintetiche", "local_real": "Prove reali locali"}.get(
                        title, title
                    ),
                    result["captured_blocks"],
                    result["labelled_exercise_blocks"],
                    result["unknown_exercise_blocks"],
                    result["abstained_blocks"],
                    result["top3_hits"],
                    result["garmin_baseline_hits"],
                    result["context_baseline_hits"],
                )
            )
            + "</tr>"
        )
    return (
        '<!doctype html><html lang="it"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Prove locali dei suggerimenti</title><style>"
        "body{font:16px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:20px;"
        "color:#20343f}table{border-collapse:collapse;width:100%}"
        "td,th{padding:10px;border:1px solid #cbd8dd}.table{overflow:auto}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        "<h1>Prove locali dei suggerimenti</h1><p>"
        + ("DATI INTERAMENTE SINTETICI" if payload.get("synthetic") else "Prove solo locali")
        + "</p><p>Alternative congelate prima della risposta; etichette dichiarate "
        "dall’utente. Ignoto non equivale a errore. Nessuna validità predittiva "
        'reale dimostrata dai test sintetici.</p><div class="table"><table><thead><tr>'
        "<th>Gruppo</th><th>Blocchi</th><th>Esercizio noto</th><th>Ignoto</th>"
        "<th>Astensioni</th><th>Riscontri top 3</th><th>Riscontri Garmin</th>"
        "<th>Riscontri solo contesto</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        "<p>I tre confronti usano lo stesso denominatore: blocchi con esercizio "
        "ricordato, comprese le astensioni. La copertura usa tutti i blocchi "
        "congelati. Le famiglie hanno un denominatore separato: risposte che "
        "dichiarano almeno una famiglia.</p>"
        "<p>Tempo in secondi e clic sono osservati solo dopo adesione facoltativa "
        "nel modulo minimo. Le revisioni superate non aumentano il campione. "
        "Non sono una prova indipendente del tempo totale necessario.</p>"
        "<details><summary>Conteggi, valori mancanti e provenienza</summary><pre>"
        + html.escape(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        + "</pre></details></html>"
    )
