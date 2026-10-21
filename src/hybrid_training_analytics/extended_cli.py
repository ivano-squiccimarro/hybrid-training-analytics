"""CLI entry points for optional analytics and local-only interface."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from . import __version__
from .advanced_metrics import DecouplingConfig, compare_phases
from .annotations import validate_annotations
from .extended_reporting import enrich, write_descriptor_report, write_extended
from .hybrid import hybrid_summary
from .ingestion import read_fit
from .local_catalog import LocalCatalog
from .models import DataError, RunData
from .reporting import new_output, write_json
from .signals import read_signals
from .strength import COLUMNS, read_strength
from .synthetic import demo_samples, fit_bytes

COMMANDS = {
    "enrich",
    "signals",
    "hybrid-report",
    "local-hybrid-report",
    "strength-evaluation-report",
    "compare-phases",
    "demo-full",
    "serve",
    "strength-fit",
    "strength-reconcile-demo",
}


def add_commands(commands) -> None:
    strength = commands.add_parser(
        "strength-fit", help="FIT/ZIP forza -> qualità, nomi e blocchi ipotizzati"
    )
    strength.add_argument("fit", type=Path)
    strength.add_argument("--out", type=Path, required=True)
    strength_demo = commands.add_parser("strength-reconcile-demo", help="Demo forza sintetica")
    strength_demo.add_argument("--out", type=Path, required=True)
    enriched = commands.add_parser(
        "enrich", help="Minuti, annotazioni, decoupling e sensibilità offline"
    )
    enriched.add_argument("run_json", type=Path)
    enriched.add_argument("--annotations", type=Path)
    enriched.add_argument("--signals", type=Path)
    enriched.add_argument("--fit-signals", type=Path)
    enriched.add_argument("--config", type=Path, help="JSON DecouplingConfig completo")
    enriched.add_argument("--out", type=Path, required=True)
    signals = commands.add_parser("signals", help="FIT -> RunData + sidecar cadenza/potenza")
    signals.add_argument("fit", type=Path)
    signals.add_argument("--out", type=Path, required=True)
    hybrid = commands.add_parser(
        "hybrid-report", help="Diario forza, RPE e contesto temporale corsa"
    )
    local_hybrid = commands.add_parser(
        "local-hybrid-report", help="Report offline del catalogo locale, incluse sedute minime"
    )
    local_hybrid.add_argument("--workspace", type=Path, default=Path("."))
    local_hybrid.add_argument("--out", type=Path, required=True)
    evaluation = commands.add_parser(
        "strength-evaluation-report", help="Prove locali facoltative: denominatori e astensione"
    )
    evaluation.add_argument("--workspace", type=Path, default=Path("."))
    evaluation.add_argument("--out", type=Path, required=True)
    hybrid.add_argument("csv", type=Path, nargs="?")
    hybrid.add_argument(
        "--strength-summaries", type=Path, help="JSON lista delle ultime sedute minime"
    )
    hybrid.add_argument("--annotations", type=Path, help="Mappa session_id -> dichiarazioni")
    hybrid.add_argument("--runs", type=Path, nargs="*", default=[])
    hybrid.add_argument("--timezone", default="Europe/Rome")
    hybrid.add_argument("--out", type=Path, required=True)
    compare = commands.add_parser("compare-phases", help="Confronto fra due analysis_extended.json")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("comparison", type=Path)
    compare.add_argument("--out", type=Path, required=True)
    demo = commands.add_parser("demo-full", help="Esempio completo con soli dati sintetici")
    demo.add_argument("--out", type=Path, required=True)
    web = commands.add_parser(
        "serve", help="Interfaccia locale su 127.0.0.1, nessun servizio esterno"
    )
    web.add_argument("--workspace", type=Path, default=Path("."))
    web.add_argument("--port", type=int, default=8765)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_payloads(a: dict, b: dict) -> dict:
    def context(payload):
        values = payload["annotations"]
        return {
            "sensor": values["sensor"],
            "incline_pct": values["incline_pct"],
            "output_basis": "uniform_scaling_estimate"
            if payload["calibration"]
            else "fit_device_channel",
        }

    return compare_phases(
        a["decoupling"], b["decoupling"], context_a=context(a), context_b=context(b)
    )


def descriptor_output(payload: dict, output: Path) -> Path:
    new_output(output)
    marker = output / "INCOMPLETE.txt"
    marker.write_text("Gerenazione in corso", encoding="utf-8")
    write_json(output / "analysis.json", payload)
    write_descriptor_report(payload, output / "report.html")
    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "versions": {"hta": __version__, "python": platform.python_version()},
            "input_sha256": payload.get("input_sha256"),
            "output_sha256": hashlib.sha256((output / "analysis.json").read_bytes()).hexdigest(),
            "report_sha256": hashlib.sha256((output / "report.html").read_bytes()).hexdigest(),
            "domain_code_sha256": {
                name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                for name in (
                    "hybrid.py",
                    "annotations.py",
                    "strength_sessions.py",
                    "strength_evaluation.py",
                    "strength_catalog.py",
                    "reconciliation.py",
                    "extended_reporting.py",
                    "local_catalog.py",
                )
            },
        },
    )
    marker.unlink()
    return (output / "report.html").resolve()


def execute_extended(args) -> Path:
    if args.command in {"strength-fit", "strength-reconcile-demo"}:
        from .strength_workflow import strength_demo, strength_output

        return (
            strength_demo(args.out)
            if args.command == "strength-reconcile-demo"
            else strength_output(args.fit, args.out)
        )
    if args.command == "serve":
        from .local_web import serve

        if not 0 <= args.port <= 65535:
            raise DataError("Porta fuori range")
        serve(args.workspace, args.port)
        return args.workspace.resolve()
    if args.command == "demo-full":
        return demo_full(args.out)
    if args.command == "signals":
        run = read_fit(args.fit)
        signals = read_signals(args.fit, run)
        new_output(args.out)
        marker = args.out / "INCOMPLETE.txt"
        marker.write_text("Segnali in corso", encoding="utf-8")
        write_json(args.out / "run.json", run.to_dict())
        write_json(args.out / "signals.json", signals)
        marker.unlink()
        return (args.out / "signals.json").resolve()
    if args.command == "enrich":
        content = args.run_json.read_bytes()
        run = RunData.from_dict(json.loads(content))
        declarations = {} if args.annotations is None else read_json(args.annotations)
        if (
            isinstance(declarations, dict)
            and declarations.get("artifact_type") == "user_annotations"
        ):
            declarations = declarations["values"]
        declarations = validate_annotations(declarations)
        if args.signals and args.fit_signals:
            raise DataError("Usare --signals oppure --fit-signals")
        signals = (
            read_json(args.signals)
            if args.signals
            else read_signals(args.fit_signals, run)
            if args.fit_signals
            else None
        )
        config = (
            DecouplingConfig(**read_json(args.config))
            if args.config
            else DecouplingConfig(
                output_confirmed=declarations["speed_confirmed"],
                constant_incline=declarations["constant_incline"],
            )
        )
        payload = enrich(run, declarations, config, signals)
        return write_extended(
            run,
            payload,
            args.out,
            input_sha256=hashlib.sha256(content).hexdigest(),
            signals=signals,
        )
    if args.command == "strength-evaluation-report":
        from .strength_evaluation import StrengthEvaluation

        return descriptor_output(StrengthEvaluation(args.workspace).metrics(), args.out)
    if args.command == "local-hybrid-report":
        payload = LocalCatalog(args.workspace).hybrid()
        return descriptor_output(payload, args.out)
    if args.command == "hybrid-report":
        declarations = {} if args.annotations is None else read_json(args.annotations)
        runs = []
        for path in args.runs:
            row = read_json(path)
            row["run_id"] = row["source_sha256"]
            runs.append(row)
        summaries = (
            []
            if args.strength_summaries is None
            else json.loads(args.strength_summaries.read_text(encoding="utf-8"))
        )
        if not isinstance(summaries, list) or (args.csv is None and not summaries):
            raise DataError("Servono un CSV oppure sedute minime come lista JSON")
        payload = hybrid_summary(
            [] if args.csv is None else read_strength(args.csv),
            declarations,
            runs,
            args.timezone,
            strength_summaries=summaries,
        )
        payload["input_sha256"] = {
            "strength_csv": hashlib.sha256(args.csv.read_bytes()).hexdigest() if args.csv else None,
            "strength_summaries": hashlib.sha256(args.strength_summaries.read_bytes()).hexdigest()
            if args.strength_summaries
            else None,
            "annotations": None
            if args.annotations is None
            else hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
            "run_analyses": [hashlib.sha256(path.read_bytes()).hexdigest() for path in args.runs],
        }
        return descriptor_output(payload, args.out)
    if args.command == "compare-phases":
        payload = compare_payloads(read_json(args.baseline), read_json(args.comparison))
        payload["artifact_type"] = "matched_phase_comparison"
        payload["input_sha256"] = {
            "baseline": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
            "comparison": hashlib.sha256(args.comparison.read_bytes()).hexdigest(),
        }
        return descriptor_output(payload, args.out)
    raise DataError("Comando non disponibile")


def demo_full(output: Path) -> Path:
    new_output(output)
    marker = output / "INCOMPLETE.txt"
    marker.write_text("Demo sintetica in corso", encoding="utf-8")
    catalog = LocalCatalog(output / "workspace")
    results = []
    keys = []
    for day in (0, 1):
        samples = [
            replace(
                s,
                timestamp_utc=(
                    datetime.fromisoformat(s.timestamp_utc) + timedelta(days=day)
                ).isoformat(),
                heart_rate_bpm=None if s.heart_rate_bpm is None else s.heart_rate_bpm + day * 2,
            )
            for s in demo_samples()
        ]
        content = fit_bytes(
            samples, signals=[{"cadence_fit_rpm": 85, "power_w": 200} for _ in samples]
        )
        key = catalog.import_fit(content, synthetic=True)
        keys.append(key)
        catalog.annotate(
            "run:" + key,
            {
                "sensor": "wrist",
                "session_rpe": 4,
                "incline_pct": 1,
                "constant_incline": True,
                "speed_confirmed": True,
                "target_distance_m": 6000,
                "distance_source": "user_other",
            },
        )
        results.append(
            catalog.analyze(
                key,
                {"start_s": 300, "end_s": 1800, "output_confirmed": True, "constant_incline": True},
            )
        )
    inputs = output / "inputs"
    inputs.mkdir()
    csv_path = inputs / "strength_synthetic.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        for day, load in ((13, 50), (14, 55)):
            for index in (1, 2, 3):
                writer.writerow(
                    [
                        f"2025-01-{day}T17:10:00+00:00",
                        f"synthetic-{day}",
                        "squat",
                        "external",
                        80,
                        0,
                        load,
                        index,
                        5,
                        2,
                        8,
                    ]
                )
    catalog.import_strength(csv_path.read_bytes())
    for day in (13, 14):
        catalog.annotate(
            "strength:" + f"synthetic-{day}",
            {
                "start_utc": f"2025-01-{day}T17:00:00+00:00",
                "end_utc": f"2025-01-{day}T18:00:00+00:00",
                "session_rpe": 6,
                "lower_body_exposure": "high",
                "technique": "synthetic_same_rom",
            },
        )
    payload = {
        "schema_version": 1,
        "artifact_type": "full_synthetic_demo",
        "synthetic": True,
        "analyses": [
            {"run_id": key, "output_id": row["output_id"]}
            for key, row in zip(keys, results, strict=True)
        ],
        "comparison": catalog.compare(keys[0], keys[1]),
        "hybrid": catalog.hybrid(),
        "local_ui_command": (
            'uv run --locked python -m hybrid_training_analytics serve --workspace "'
        )
        + str(catalog.workspace)
        + '" --port 8765',
    }
    write_json(output / "demo.json", payload)
    write_descriptor_report(payload, output / "report.html")
    marker.unlink()
    return (output / "report.html").resolve()
