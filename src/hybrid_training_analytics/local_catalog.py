"""Private local catalog: content identities, append-only annotations and output revisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .advanced_metrics import DecouplingConfig, compare_phases
from .annotations import append_revision, revision_key, validate_annotations
from .extended_reporting import enrich, write_extended
from .hybrid import hybrid_summary
from .ingestion import read_fit
from .models import DataError, RunData
from .reporting import write_json
from .signals import read_signals
from .strength import read_strength, summarize_strength


def identity(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise DataError("Identificatore locale non valido")
    return value


class LocalCatalog:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.root = self.workspace / "data" / "processed" / "hta-local"
        self.reports = self.workspace / "reports"

    def import_fit(self, content: bytes, *, synthetic: bool = False) -> str:
        if not content or len(content) > 10 * 1024 * 1024:
            raise DataError("FIT vuoto o oltre limite tecnico 10 MiB")
        key = hashlib.sha256(content).hexdigest()
        directory = self.root / "runs" / key
        if directory.exists():
            self.load_run(key)
            return key
        # Decoder validates before creating a durable catalog entry.
        scratch = self.workspace / "work" / ("full-import-" + uuid4().hex)
        scratch.mkdir(parents=True)
        path = scratch / "input.fit"
        path.write_bytes(content)
        run = read_fit(path, synthetic=synthetic)
        signals = read_signals(path, run)
        directory.mkdir(parents=True, exist_ok=False)
        marker = directory / "INCOMPLETE.txt"
        marker.write_text("Importazione locale in corso", encoding="utf-8")
        (directory / "source.fit").write_bytes(content)
        write_json(directory / "run.json", run.to_dict())
        write_json(directory / "signals.json", signals)
        write_json(
            directory / "import.json",
            {"schema_version": 1, "source_sha256": key, "synthetic": synthetic},
        )
        marker.unlink()
        return key

    def load_run(self, key: str) -> tuple[RunData, dict]:
        directory = self.root / "runs" / identity(key)
        if (directory / "INCOMPLETE.txt").exists():
            raise DataError("Importazione precedente incompleta: conservata per diagnosi")
        run = RunData.from_dict(json.loads((directory / "run.json").read_text(encoding="utf-8")))
        if run.source_sha256 != key:
            raise DataError("IdentitÃƒÂ  del catalogo incompatibile")
        signals = json.loads((directory / "signals.json").read_text(encoding="utf-8"))
        return run, signals

    def annotation_directory(self, subject: str) -> Path:
        key = hashlib.sha256(subject.encode()).hexdigest()
        return self.root / "annotations" / key

    def annotations(self, subject: str) -> dict:
        paths = list(self.annotation_directory(subject).glob("*.json"))
        if not paths:
            return validate_annotations({})
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        last = max(rows, key=revision_key)
        return validate_annotations(last["values"])

    def annotate(self, subject: str, values: dict) -> dict:
        if subject.startswith("run:"):
            self.load_run(subject[4:])
        elif subject.startswith("strength:"):
            sets = [r for r in self.strength_sets() if r.session_id == subject[9:]]
            if not sets:
                raise DataError("Sessione forza sconosciuta")
            clean = validate_annotations(values)
            if clean["start_utc"] and any(
                not clean["start_utc"] <= r.timestamp_utc <= clean["end_utc"] for r in sets
            ):
                raise DataError("Serie forza fuori intervallo dichiarato")
        else:
            raise DataError("Tipo annotazione sconosciuto")
        return append_revision(self.annotation_directory(subject), subject, values)

    def runs(self) -> list[dict]:
        result = []
        for path in (self.root / "runs").glob("*"):
            if not path.is_dir() or (path / "INCOMPLETE.txt").exists():
                continue
            run, _ = self.load_run(path.name)
            result.append(
                {
                    "run_id": path.name,
                    "start_utc": run.samples[0].timestamp_utc,
                    "duration_s": run.samples[-1].elapsed_s,
                    "record_distance_delta_m": None
                    if run.samples[0].distance_m is None or run.samples[-1].distance_m is None
                    else run.samples[-1].distance_m - run.samples[0].distance_m,
                    "records": len(run.samples),
                    "synthetic": run.synthetic,
                    "annotations": self.annotations("run:" + path.name),
                }
            )
        return sorted(result, key=lambda r: r["start_utc"], reverse=True)

    def analyze(self, key: str, options: dict) -> dict:
        run, signals = self.load_run(key)
        try:
            config = DecouplingConfig(**options)
        except TypeError as exc:
            raise DataError("Campi configurazione decoupling sconosciuti") from exc
        values = self.annotations("run:" + key)
        payload = enrich(run, values, config, signals)
        output = self.reports / ("full-" + key + "-" + uuid4().hex)
        run_path = self.root / "runs" / key / "run.json"
        write_extended(
            run,
            payload,
            output,
            input_sha256=hashlib.sha256(run_path.read_bytes()).hexdigest(),
            signals=signals,
        )
        refs = self.root / "runs" / key / "analyses"
        refs.mkdir(exist_ok=True)
        write_json(refs / (output.name + ".json"), {"output_directory": output.name})
        return {"output_id": output.name, "analysis": payload}

    def latest_analysis(self, key: str) -> dict | None:
        self.load_run(key)
        refs = list((self.root / "runs" / key / "analyses").glob("*.json"))
        if not refs:
            return None
        latest = max(refs, key=lambda p: p.stat().st_mtime_ns)
        name = json.loads(latest.read_text(encoding="utf-8"))["output_directory"]
        path = self.output_file(name, "analysis_extended.json")
        return json.loads(path.read_text(encoding="utf-8"))

    def output_file(self, output_id: str, filename: str) -> Path:
        if (
            not isinstance(output_id, str)
            or not output_id.startswith("full-")
            or any(c not in "full-0123456789abcdef" for c in output_id)
            or len(output_id) != 102
        ):
            raise DataError("Output locale non valido")
        if filename not in {"minutes.csv", "analysis_extended.json", "manifest.json"}:
            raise DataError("Artefatto non esportabile da questa rotta")
        directory = self.reports / output_id
        if (directory / "INCOMPLETE.txt").exists():
            raise DataError("Output incompleto")
        return directory / filename

    def import_strength(self, content: bytes) -> list[dict]:
        if not content or len(content) > 1024 * 1024:
            raise DataError("CSV vuoto o oltre limite tecnico 1 MiB")
        key = hashlib.sha256(content).hexdigest()
        folder = self.root / "strength" / key
        if folder.exists():
            return summarize_strength(read_strength(folder / "sets.csv"))
        scratch = self.workspace / "work" / ("full-strength-" + uuid4().hex)
        scratch.mkdir(parents=True)
        path = scratch / "sets.csv"
        path.write_bytes(content)
        rows = read_strength(path)
        existing = {r.session_id for r in self.strength_sets()}
        if any(r.session_id in existing for r in rows):
            raise DataError("Sessione forza giÃƒÂ  importata: nessuna sovrascrittura")
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "sets.csv").write_bytes(content)
        return summarize_strength(rows)

    def strength_sets(self) -> list:
        return [
            r
            for path in sorted((self.root / "strength").glob("*/sets.csv"))
            for r in read_strength(path)
        ]

    def hybrid(self) -> dict:
        sets = self.strength_sets()
        declarations = {r.session_id: self.annotations("strength:" + r.session_id) for r in sets}
        runs = []
        for row in self.runs():
            latest = self.latest_analysis(row["run_id"])
            row["cardiovascular_exposure"] = (
                None if latest is None else latest["cardiovascular_exposure"]
            )
            runs.append(row)
        from .strength_sessions import StrengthSessions

        summaries = StrengthSessions(self.workspace).sessions()
        result = hybrid_summary(sets, declarations, runs, strength_summaries=summaries)
        result["input_sha256"] = {
            name: hashlib.sha256(
                json.dumps(value, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            for name, value in (
                ("known_strength_sets", [asdict(r) for r in sets]),
                ("strength_annotations", declarations),
                ("minimal_sessions", summaries),
                ("runs_with_annotations_and_exposure", runs),
            )
        }
        return result

    def compare(self, baseline: str, comparison: str) -> dict:
        a, b = self.latest_analysis(baseline), self.latest_analysis(comparison)
        if a is None or b is None:
            raise DataError("Analizzare entrambe le corse prima del confronto")

        def context(payload: dict) -> dict:
            values = payload["annotations"]
            return {
                "sensor": values["sensor"],
                "incline_pct": values["incline_pct"],
                "output_basis": "uniform_scaling_estimate"
                if payload["calibration"]
                else "fit_device_channel",
            }

        result = compare_phases(
            a["decoupling"], b["decoupling"], context_a=context(a), context_b=context(b)
        )
        result["baseline_run_id"] = baseline
        result["comparison_run_id"] = comparison
        result["concurrent_contexts"] = [
            row for row in self.hybrid()["run_contexts"] if row["run_id"] in {baseline, comparison}
        ]
        return result
