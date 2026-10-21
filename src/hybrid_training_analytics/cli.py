"""Local CLI. Inputs validated before reserving a fresh output directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from .calibration import CalibrationConfig, calibrate
from .ingestion import read_fit
from .metrics import AnalysisConfig, TreadmillSegment, analyze, read_treadmill
from .models import DataError, RunData
from .strength import COLUMNS, read_strength, summarize_strength
from .synthetic import demo_samples, fit_bytes


def _analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--speed-reliable",
        action="store_true",
        help="Velocità Garmin verificata dall'utente; nessuna calibrazione implicita",
    )
    parser.add_argument("--treadmill-segments", type=Path, help="Log manuale tapis roulant CSV")
    parser.add_argument(
        "--constant-incline",
        action="store_true",
        help="Pendenza verificata costante nella finestra di analisi",
    )
    parser.add_argument("--warmup-s", type=float, default=300)
    parser.add_argument("--segment-start-s", type=float)
    parser.add_argument("--segment-end-s", type=float)
    parser.add_argument("--rpe", type=float, help="RPE dichiarata della corsa, 1..10")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="hta", description="Analisi allenamenti locale e prudente"
    )
    commands = result.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import", help="FIT -> normalizzazione + report in un comando")
    importer.add_argument("fit", type=Path)
    importer.add_argument("--timezone", default="Europe/Rome")
    importer.add_argument("--out", type=Path, required=True)
    _analysis_arguments(importer)
    report = commands.add_parser("report", help="Riproduci report da run.json normalizzato")
    report.add_argument("run_json", type=Path)
    report.add_argument("--out", type=Path, required=True)
    report.add_argument(
        "--config", type=Path, help="analysis.json originale: ripristina tutti i parametri"
    )
    _analysis_arguments(report)
    demo = commands.add_parser("demo", help="Genera FIT, diario e report interamente sintetici")
    demo.add_argument("--out", type=Path, required=True)
    strength = commands.add_parser(
        "strength-report", help="Valida diario CSV e descrivi volume/serie"
    )
    strength.add_argument("csv", type=Path)
    strength.add_argument("--out", type=Path, required=True)
    calibration = commands.add_parser("calibrate", help="Derivato proporzionale da run.json")
    calibration.add_argument("run_json", type=Path)
    calibration.add_argument("--distance-m", type=float, required=True)
    calibration.add_argument(
        "--distance-source", choices=["watch_final", "treadmill_log", "user_other"], required=True
    )
    calibration.add_argument("--incline-pct", type=float, help="Pendenza tapis dichiarata costante")
    calibration.add_argument("--max-gap-s", type=float, default=10)
    calibration.add_argument("--out", type=Path, required=True)
    from .extended_cli import add_commands

    add_commands(commands)
    return result


def _config(args: argparse.Namespace) -> tuple[AnalysisConfig, list[TreadmillSegment]]:
    if getattr(args, "config", None):
        if (
            args.speed_reliable
            or args.constant_incline
            or args.treadmill_segments
            or args.warmup_s != 300
            or args.segment_start_s is not None
            or args.segment_end_s is not None
            or args.rpe is not None
        ):
            raise DataError("Con --config non combinare override di analisi")
        payload = json.loads(args.config.read_text(encoding="utf-8"))
        try:
            config = AnalysisConfig(**payload["config"])
            treadmill = [TreadmillSegment(**row) for row in payload["treadmill_segments"]]
        except (KeyError, TypeError) as exc:
            raise DataError(f"Configurazione non valida: {exc}") from exc
    else:
        config = AnalysisConfig(
            speed_reliable=args.speed_reliable,
            constant_incline=args.constant_incline,
            warmup_s=args.warmup_s,
            segment_start_s=args.segment_start_s,
            segment_end_s=args.segment_end_s,
            rpe=args.rpe,
        )
        treadmill = read_treadmill(args.treadmill_segments) if args.treadmill_segments else []
    config.validate()
    return config, treadmill


def _write_demo_inputs(output: Path) -> tuple[Path, Path, Path]:
    inputs = output / "inputs"
    inputs.mkdir()
    fit = inputs / "synthetic.fit"
    fit.write_bytes(fit_bytes(demo_samples()))
    treadmill = inputs / "treadmill_synthetic.csv"
    treadmill.write_text("start_s,end_s,speed_kmh,incline_pct\n0,1800,12,1\n", encoding="utf-8")
    strength = inputs / "strength_synthetic.csv"
    with strength.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        for day, load in [(6, 10), (13, 12)]:
            for index in range(1, 4):
                writer.writerow(
                    [
                        f"2025-01-{day:02d}T18:00:00+01:00",
                        f"synthetic-{day}",
                        "dip",
                        "bodyweight",
                        80,
                        load,
                        0,
                        index,
                        6,
                        2,
                        8,
                    ]
                )
    return fit, treadmill, strength


def execute(args: argparse.Namespace) -> Path:
    from .extended_cli import COMMANDS, execute_extended
    from .reporting import new_output, write_run_report, write_strength_report

    if args.command in COMMANDS:
        return execute_extended(args)

    if args.command == "calibrate":
        from .calibration_io import write_calibration

        source_bytes = args.run_json.read_bytes()
        run = RunData.from_dict(json.loads(source_bytes))
        payload = calibrate(
            run,
            CalibrationConfig(
                args.distance_m, args.distance_source, args.incline_pct, args.max_gap_s
            ),
        )
        new_output(args.out)
        marker = args.out / "INCOMPLETE.txt"
        marker.write_text("Derivato proporzionale in corso.\n", encoding="utf-8")
        write_calibration(payload, args.out, hashlib.sha256(source_bytes).hexdigest())
        marker.unlink()
        return (args.out / "calibration.json").resolve()
    if args.command == "strength-report":
        summary = summarize_strength(read_strength(args.csv))
        new_output(args.out)
        marker = args.out / "INCOMPLETE.txt"
        marker.write_text("Output incompleto se questo file rimane presente.\n", encoding="utf-8")
        write_strength_report(summary, args.out)
    elif args.command == "demo":
        new_output(args.out)
        marker = args.out / "INCOMPLETE.txt"
        marker.write_text("Generazione demo in corso.\n", encoding="utf-8")
        fit, treadmill_path, strength_path = _write_demo_inputs(args.out)
        run = read_fit(fit, synthetic=True)
        analysis = analyze(
            run, AnalysisConfig(speed_reliable=True, rpe=6), read_treadmill(treadmill_path)
        )
        write_run_report(run, analysis, args.out)
        strength_output = args.out / "strength"
        new_output(strength_output)
        write_strength_report(
            summarize_strength(read_strength(strength_path)), strength_output, synthetic=True
        )
    else:
        config, treadmill = _config(args)
        run = (
            read_fit(args.fit, timezone=args.timezone)
            if args.command == "import"
            else (RunData.from_dict(json.loads(args.run_json.read_text(encoding="utf-8"))))
        )
        analysis = analyze(run, config, treadmill)
        new_output(args.out)
        marker = args.out / "INCOMPLETE.txt"
        marker.write_text("Report in corso.\n", encoding="utf-8")
        write_run_report(run, analysis, args.out)
    marker.unlink()  # This invocation created the marker; no user output is deleted.
    return (args.out / "report.html").resolve()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = execute(args)
    except (DataError, OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
        print(f"Errore: {exc}", file=sys.stderr)
        return 2
    if args.command == "serve":
        return 0
    label = "Derivato completato" if args.command == "calibrate" else "Report completato"
    print(f"{label}: {report}")
    return 0
