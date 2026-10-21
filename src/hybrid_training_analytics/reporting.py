"""Offline self-contained HTML, Markdown and strict JSON; no CDN or telemetry."""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import os
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .models import DataError, RunData  # noqa: E402


def new_output(path: Path) -> None:
    """Reserve a new directory. Never overwrite existing or expensive user output."""
    if path.exists():
        raise DataError(f"Output già esistente: {path}. Scegliere una nuova cartella.")
    path.mkdir(parents=True)


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def _run_chart(run: RunData, speed_reliable: bool, max_gap_s: float = 10) -> str:
    fig, axes = plt.subplots(2 if speed_reliable else 1, 1, figsize=(10, 5), squeeze=False)
    times, heart_rates, paces = [], [], []
    previous = None
    for sample in run.samples:
        if previous is not None and sample.elapsed_s - previous > max_gap_s:
            times.append(sample.elapsed_s / 60)
            heart_rates.append(float("nan"))
            paces.append(float("nan"))
        times.append(sample.elapsed_s / 60)
        heart_rates.append(
            sample.heart_rate_bpm if sample.heart_rate_bpm is not None else float("nan")
        )
        paces.append(1000 / sample.speed_m_s / 60 if sample.speed_m_s else float("nan"))
        previous = sample.elapsed_s
    axes[0, 0].plot(
        times,
        heart_rates,
        color="#0f766e",
        linewidth=1.4,
    )
    axes[0, 0].set(ylabel="FC registrata (bpm)", xlabel="Tempo trascorso (min)")
    axes[0, 0].grid(alpha=0.15)
    for start, end in run.pauses_s:
        axes[0, 0].axvspan(start / 60, end / 60, color="#d97706", alpha=0.15)
    if speed_reliable:
        axes[1, 0].plot(
            times,
            paces,
            color="#2563eb",
        )
        axes[1, 0].set(ylabel="Passo stimato (min/km)", xlabel="Tempo trascorso (min)")
        axes[1, 0].grid(alpha=0.15)
    fig.suptitle("DATI SINTETICI — dimostrazione" if run.synthetic else "Serie registrate nel FIT")
    fig.tight_layout()
    stream = io.StringIO()
    with matplotlib.rc_context({"svg.hashsalt": run.source_sha256}):
        fig.savefig(stream, format="svg", metadata={"Date": None}, bbox_inches="tight")
    plt.close(fig)
    content = stream.getvalue()
    return content[content.index("<svg") :]


def _value(value: object, unit: str = "") -> str:
    if value is None:
        return "non disponibile"
    return f"{value:.2f} {unit}".strip() if isinstance(value, float) else str(value)


STYLE = """
body{font:16px/1.6 system-ui,sans-serif;color:#1e293b;background:#f8fafc;margin:0}
main{max-width:1060px;margin:36px auto;padding:32px;background:white;border-radius:16px}
h1{line-height:1.2;color:#0f766e}h2{margin-top:32px}table{border-collapse:collapse;width:100%}
td,th{padding:10px;text-align:left;border-bottom:1px solid #e2e8f0}svg{max-width:100%;height:auto}
.badge{background:#fef3c7;padding:8px 14px;border-radius:8px;font-weight:700}
code{overflow-wrap:anywhere}pre{white-space:pre-wrap}.muted{color:#64748b}
@media(max-width:700px){main{margin:8px;padding:18px}table{font-size:13px}}
"""


def write_run_report(run: RunData, analysis: dict, output: Path) -> None:
    obs, quality = analysis["observations"], analysis["quality"]
    drift, comparison = analysis["drift"], analysis["comparison"]
    local = datetime.fromisoformat(analysis["start_utc"]).astimezone(ZoneInfo(run.timezone))
    rows = [
        ("Inizio locale (offset esplicito)", local.isoformat()),
        ("Fuso display", run.timezone),
        ("Durata trascorsa registrata", _value(obs["elapsed_duration_s"], "s")),
        (
            "Distanza tra record Garmin — stima indoor",
            _value(obs["garmin_record_distance_delta_m"], "m"),
        ),
        (
            "FC media pesata sul tempo valido — osservazione",
            _value(obs["heart_rate_time_weighted_bpm"], "bpm"),
        ),
        ("Copertura temporale FC valida", _value(100 * quality["hr_time_coverage_fraction"], "%")),
        ("RPE dichiarata (1..10)", _value(obs["rpe_self_report_1_10"])),
        (
            "Passo da velocità confermata — stima",
            _value(analysis["estimates"]["pace_s_km_user_confirmed_speed"], "s/km"),
        ),
    ]
    if drift["status"] == "available":
        reference_label = (
            "log manuale tapis roulant"
            if drift["reference"] == "treadmill_setting_user_log"
            else "velocità Garmin confermata dall'utente"
        )
        drift_text = (
            f"Variazione FC: {drift['heart_rate_change_pct']:.2f}% tra le due metà della "
            f"finestra {drift['start_s']:.0f}–{drift['end_s']:.0f} s. "
            f"Riferimento: {reference_label}. {drift['note']}"
        )
    else:
        drift_text = "Non disponibile: " + "; ".join(drift["reasons"])
    if comparison["status"] == "available":
        comparison_text = (
            f"Garmin {comparison['garmin_record_distance_m']:.0f} m; distanza integrata dalle "
            f"impostazioni tapis {comparison['treadmill_setting_distance_m']:.0f} m; "
            f"differenza {comparison['difference_pct']:.2f}%. {comparison['note']}"
        )
    else:
        comparison_text = "Non disponibile: " + comparison["reason"]
    chart = _run_chart(run, analysis["config"]["speed_reliable"], analysis["config"]["max_gap_s"])
    (output / "timeseries.svg").write_text(chart, encoding="utf-8")
    title = "Report corsa — DATI SINTETICI" if run.synthetic else "Report corsa locale"
    warning_items = "".join(f"<li>{html.escape(w)}</li>" for w in analysis["warnings"])
    limit_items = "".join(f"<li>{html.escape(w)}</li>" for w in analysis["limits"])
    table = "".join(f"<tr><th>{html.escape(a)}</th><td>{html.escape(b)}</td></tr>" for a, b in rows)
    badge = (
        '<p class="badge">Dati interamente inventati; nessuna valutazione dell’atleta.</p>'
        if (run.synthetic)
        else '<p class="badge">Report locale: contiene dati personali, non pubblicare.</p>'
    )
    content = (
        f'<!doctype html><html lang="it"><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{STYLE}</style><main>"
        f"<h1>{html.escape(title)}</h1>{badge}<table>{table}</table>"
        f"<h2>FC nel tempo e passo disponibile</h2>{chart}"
        f"<h2>Deriva descrittiva — inferenza</h2><p>{html.escape(drift_text)}</p>"
        f"<h2>Confronto Garmin / tapis roulant</h2><p>{html.escape(comparison_text)}</p>"
        f"<h2>Qualità e controlli</h2><ul>{warning_items}</ul>"
        f"<p>Accuratezza del sensore: sconosciuta. La FC registrata può contenere artefatti.</p>"
        f"<h2>Limiti</h2><ul>{limit_items}</ul>"
        f'<p class="muted">SHA-256 sorgente: <code>{run.source_sha256}</code>. '
        f"Parametri e versioni software: analysis.json e manifest.json.</p></main></html>"
    )
    (output / "report.html").write_text(content, encoding="utf-8")
    markdown = [f"# {title}", "", "| Campo | Valore |", "|---|---|"]
    markdown.extend(f"| {a} | {b} |" for a, b in rows)
    encoded = base64.b64encode(chart.encode()).decode()
    markdown.extend(
        [
            "",
            f"![Serie temporale](data:image/svg+xml;base64,{encoded})",
            "",
            "## Deriva descrittiva — inferenza",
            "",
            drift_text,
            "",
            "## Confronto Garmin / tapis",
            "",
            comparison_text,
            "",
            "## Qualità e limiti",
            "",
        ]
    )
    markdown.extend(f"- {w}" for w in analysis["warnings"] + analysis["limits"])
    markdown.extend(["", f"SHA-256 sorgente: `{run.source_sha256}`", ""])
    (output / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    write_json(output / "run.json", run.to_dict())
    write_json(output / "analysis.json", analysis)
    with (output / "records.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp_utc", "elapsed_s", "heart_rate_bpm", "speed_m_s", "distance_m"])
        for sample in run.samples:
            writer.writerow(
                [
                    sample.timestamp_utc,
                    sample.elapsed_s,
                    sample.heart_rate_bpm,
                    sample.speed_m_s,
                    sample.distance_m,
                ]
            )
    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "source_sha256": run.source_sha256,
            "synthetic": run.synthetic,
            "versions": {
                name: version(name)
                for name in (
                    "hybrid-training-analytics",
                    "fitdecode",
                    "matplotlib",
                    "tzdata",
                )
            },
            "analysis_config": analysis["config"],
            "treadmill_segments": analysis["treadmill_segments"],
            "reproduction": "hta report run.json con gli stessi parametri in analysis.json",
        },
    )


def write_strength_report(summary: list[dict], output: Path, *, synthetic: bool = False) -> None:
    title = "Diario forza — DATI SINTETICI" if synthetic else "Diario forza locale"
    columns = [
        "timestamp_utc",
        "session_id",
        "exercise",
        "load_mode",
        "sets",
        "reps",
        "external_volume_kg_reps",
        "system_mass_proxy_kg_reps",
        "best_observed_load_kg",
        "best_observed_reps",
        "best_observed_rir",
        "body_mass_kg",
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    for exercise in sorted({row["exercise"] for row in summary}):
        selected = [row for row in summary if row["exercise"] == exercise]
        ax.plot(
            [datetime.fromisoformat(row["timestamp_utc"]) for row in selected],
            [row["external_volume_kg_reps"] for row in selected],
            marker="o",
            label=exercise,
        )
    ax.set(ylabel="Volume carico esterno (kg × ripetizioni)", xlabel="Data UTC")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    chart_buffer = io.StringIO()
    fig.savefig(chart_buffer, format="svg", metadata={"Date": None})
    plt.close(fig)
    chart = chart_buffer.getvalue()
    chart = chart[chart.index("<svg") :]
    head = "".join(f"<th>{c}</th>" for c in columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_value(row[c]))}</td>" for c in columns) + "</tr>"
        for row in summary
    )
    note = (
        "Serie e ripetizioni sono osservazioni; RIR/RPE sono dichiarazioni soggettive. "
        "Volume esterno e massa di sistema sono proxy, non lavoro meccanico né forza. "
        "Il miglior set osservato non è un massimale. Confrontare stesso esercizio, "
        "tecnica, ripetizioni e RIR; nessuna conversione automatica in 1RM."
    )
    content = (
        f'<!doctype html><html lang="it"><meta charset="utf-8"><title>{title}</title>'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<style>{STYLE}</style><main><h1>{title}</h1><p>{note}</p>{chart}"
        f'<div style="overflow-x:auto"><table><tr>{head}</tr>{rows}</table></div>'
        "</main></html>"
    )
    (output / "report.html").write_text(content, encoding="utf-8")
    write_json(output / "strength_summary.json", summary)
