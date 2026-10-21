"""Decode FIT strictly, then retain only the minimal running time series."""

from __future__ import annotations

import hashlib
import io
from datetime import datetime
from pathlib import Path

import fitdecode

from .models import DataError, RunData, Sample, check_timezone, number, utc_timestamp, validate_run


def normalize_records(
    records: list[dict],
    *,
    digest: str,
    timezone: str = "Europe/Rome",
    synthetic: bool = False,
    events: list[tuple[str, str]] | None = None,
) -> RunData:
    check_timezone(timezone)
    warnings: list[str] = []
    indexed: dict[str, dict] = {}
    previous_timestamp = ""
    for record in records:
        try:
            stamp = utc_timestamp(record.get("timestamp"), fit_utc=True)
        except DataError:
            warnings.append("Record senza timestamp ignorato")
            continue
        if stamp < previous_timestamp:
            warnings.append("Record fuori ordine riordinati in UTC")
        previous_timestamp = stamp
        if stamp in indexed:
            warnings.append("Timestamp duplicato: conservato ultimo record, nessuna media")
        indexed[stamp] = record
    if len(indexed) < 2:
        raise DataError("FIT senza abbastanza record temporali utilizzabili")
    origin = datetime.fromisoformat(min(indexed))
    samples = []
    previous_distance = -1.0
    for stamp, record in sorted(indexed.items()):
        hr = number(record.get("heart_rate"), minimum=1, maximum=254)
        # Prefer enhanced_speed only if present and valid; never infer speed from HR.
        speed = number(record.get("enhanced_speed"), maximum=30)
        if speed is None:
            speed = number(record.get("speed"), maximum=30)
        distance = number(record.get("distance"))
        for field, cleaned in [("heart_rate", hr), ("distance", distance)]:
            if record.get(field) is not None and cleaned is None:
                warnings.append(f"Valore non valido {field}: sostituito con null")
        if (
            record.get("speed") is not None or record.get("enhanced_speed") is not None
        ) and speed is None:
            warnings.append("Velocità non valida: sostituita con null")
        if distance is not None:
            if distance < previous_distance:
                warnings.append("Distanza cumulativa decrescente: record sostituito con null")
                distance = None
            else:
                previous_distance = distance
        samples.append(
            Sample(
                stamp,
                (datetime.fromisoformat(stamp) - origin).total_seconds(),
                hr,
                speed,
                distance,
            )
        )
    end_s = samples[-1].elapsed_s
    pauses = []
    stopped_at = None
    for stamp, kind in sorted(events or []):
        elapsed = max(0.0, (datetime.fromisoformat(stamp) - origin).total_seconds())
        if kind in {"stop", "stop_all", "stop_disable", "stop_disable_all"}:
            if stopped_at is None:
                stopped_at = elapsed
        elif kind == "start" and stopped_at is not None:
            if stopped_at < min(elapsed, end_s):
                pauses.append((stopped_at, min(elapsed, end_s)))
            stopped_at = None
    if stopped_at is not None and stopped_at < end_s:
        pauses.append((stopped_at, end_s))
    if pauses:
        warnings.append("Eventi timer stop/start rilevati; deriva vietata nelle finestre con pause")
    run = RunData(samples, digest, timezone, synthetic, sorted(set(warnings)), pauses)
    validate_run(run)
    return run


def read_fit(path: Path, *, timezone: str = "Europe/Rome", synthetic: bool = False) -> RunData:
    """Read one running activity. CRC/decoding failures never produce partial success."""
    check_timezone(timezone)
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    records: list[dict] = []
    events = []
    headers = sessions = 0
    sport = None
    try:
        with fitdecode.FitReader(
            io.BytesIO(content),
            check_crc=fitdecode.CrcCheck.RAISE,
            error_handling=fitdecode.ErrorHandling.RAISE,
        ) as reader:
            for frame in reader:
                if frame.frame_type == fitdecode.FIT_FRAME_HEADER:
                    headers += 1
                if frame.frame_type != fitdecode.FIT_FRAME_DATA:
                    continue
                # Whitelist: no GPS, serial numbers, user profiles or medical fields.
                if frame.name == "record":
                    records.append(
                        {
                            key: frame.get_value(key, fallback=None)
                            for key in (
                                "timestamp",
                                "heart_rate",
                                "speed",
                                "enhanced_speed",
                                "distance",
                            )
                        }
                    )
                elif frame.name == "session":
                    sessions += 1
                    sport = frame.get_value("sport", fallback=None)
                elif frame.name == "event" and frame.get_value("event", fallback=None) == "timer":
                    stamp = frame.get_value("timestamp", fallback=None)
                    if stamp is not None:
                        events.append(
                            (
                                utc_timestamp(stamp, fit_utc=True),
                                frame.get_value("event_type", fallback=None),
                            )
                        )
    except (fitdecode.FitError, ValueError, KeyError, EOFError) as exc:
        raise DataError(f"FIT non leggibile o CRC non valido: {exc}") from exc
    if headers != 1 or sessions > 1:
        raise DataError("FIT concatenati/multisport non supportati: esportare una singola corsa")
    if sport is not None and sport != "running":
        raise DataError(f"Attività non di corsa: {sport}")
    run = normalize_records(
        records, digest=digest, timezone=timezone, synthetic=synthetic, events=events
    )
    if sport is None:
        run.warnings.append("Sport non dichiarato: verificare che il FIT sia una corsa")
    return run
