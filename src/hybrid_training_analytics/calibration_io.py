"""Write a separate local derivative with original values and provenance."""

from __future__ import annotations

import csv
from importlib.metadata import version
from pathlib import Path

from .reporting import write_json


def write_calibration(payload: dict, output: Path, source_run_sha256: str) -> None:
    write_json(output / "calibration.json", payload)
    with (output / "records_scaled.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(payload["samples"][0]))
        writer.writeheader()
        writer.writerows(payload["samples"])
    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": payload["artifact_type"],
            "source_fit_sha256": payload["source_sha256"],
            "source_run_json_sha256": source_run_sha256,
            "algorithm_version": payload["algorithm_version"],
            "config": payload["config"],
            "synthetic": payload["synthetic"],
            "software_version": version("hybrid-training-analytics"),
        },
    )
    (output / "README.md").write_text(
        "# Traccia proporzionalmente riscalata — stime\n\n"
        + (
            "DATI SINTETICI.\n\n"
            if payload["synthetic"]
            else "Dati personali locali: non pubblicare.\n\n"
        )
        + "Record originali e stime restano in colonne separate. Le celle vuote sono null.\n"
        + "Timestamp UTC e pause invariati. Nessun campione interpolato o aggiunto.\n"
        + "Passo da velocità dispositivo e velocità da incrementi: stime distinte.\n"
        + "Errore uniforme assunto; distanza e pendenza sono dichiarazioni dell’utente.\n"
        + "GAP opzionale: modello di costo Minetti, non passo calibrato dell’atleta.\n"
        + "Provenienza, formula, configurazione e limiti: calibration.json e manifest.json.\n",
        encoding="utf-8",
    )
