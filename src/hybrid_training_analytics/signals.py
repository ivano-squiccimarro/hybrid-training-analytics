"""Optional whitelist sidecar: native FIT cadence rpm and recorded power W."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import fitdecode

from .ingestion import read_fit
from .models import DataError, RunData, number, utc_timestamp


def read_signals(path: Path, run: RunData) -> dict:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != run.source_sha256:
        raise DataError("Segnali e RunData devono provenire dagli stessi byte FIT")
    # Also validate activity kind, concatenation, timestamps and CRC.
    verified = read_fit(path, timezone=run.timezone, synthetic=run.synthetic)
    if [s.timestamp_utc for s in verified.samples] != [s.timestamp_utc for s in run.samples]:
        raise DataError("Allineamento timestamp FIT/RunData non valido")
    indexed = {}
    try:
        with fitdecode.FitReader(
            io.BytesIO(content),
            check_crc=fitdecode.CrcCheck.RAISE,
            error_handling=fitdecode.ErrorHandling.RAISE,
        ) as reader:
            for frame in reader:
                if frame.frame_type != fitdecode.FIT_FRAME_DATA or frame.name != "record":
                    continue
                stamp = frame.get_value("timestamp", fallback=None)
                if stamp is None:
                    continue
                cadence = number(frame.get_value("cadence", fallback=None), maximum=254)
                fraction = number(frame.get_value("fractional_cadence", fallback=None), maximum=1)
                if cadence is not None and fraction is not None:
                    cadence += fraction
                indexed[utc_timestamp(stamp, fit_utc=True)] = {
                    "cadence_fit_rpm": cadence,
                    "power_w": number(frame.get_value("power", fallback=None), maximum=65534),
                }
    except (fitdecode.FitError, ValueError, KeyError, EOFError) as exc:
        raise DataError(f"Segnali FIT non leggibili: {exc}") from exc
    return {
        "schema_version": 1,
        "artifact_type": "fit_signals",
        "source_sha256": run.source_sha256,
        "provenance": {
            "cadence_fit_rpm": (
                "FIT cadence + optional fractional_cadence, native rpm; no steps conversion"
            ),
            "power_w": "FIT recorded power; device algorithm not independently validated",
        },
        "samples": [
            {
                "timestamp_utc": s.timestamp_utc,
                **indexed.get(s.timestamp_utc, {"cadence_fit_rpm": None, "power_w": None}),
            }
            for s in run.samples
        ],
    }


def validate_signals(run: RunData, payload: dict | None) -> list[dict]:
    if payload is None:
        return [{"cadence_fit_rpm": None, "power_w": None} for _ in run.samples]
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("artifact_type") != "fit_signals"
        or payload.get("source_sha256") != run.source_sha256
    ):
        raise DataError("Sidecar segnali: schema o hash incompatibile")
    rows = payload.get("samples")
    if not isinstance(rows, list) or len(rows) != len(run.samples):
        raise DataError("Sidecar segnali: numero record incompatibile")
    for sample, row in zip(run.samples, rows, strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != {"timestamp_utc", "cadence_fit_rpm", "power_w"}
            or row.get("timestamp_utc") != sample.timestamp_utc
        ):
            raise DataError("Sidecar segnali: timestamp non allineato")
        for key, limit in (("cadence_fit_rpm", 255), ("power_w", 65534)):
            if key not in row or (
                row[key] is not None
                and (
                    not isinstance(row[key], (int, float))
                    or number(row[key], maximum=limit) is None
                )
            ):
                raise DataError(f"Sidecar segnali: {key} non valido")
    return rows
