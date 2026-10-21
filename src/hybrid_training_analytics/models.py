"""Small versioned contract: SI units, UTC instants, no athlete identifiers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class DataError(ValueError):
    """Invalid input that the CLI can explain without a traceback."""


def number(value: Any, *, minimum: float = 0, maximum: float = math.inf) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and minimum <= result <= maximum else None


def check_timezone(name: str) -> None:
    if not isinstance(name, str):
        raise DataError("Fuso IANA deve essere una stringa")
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise DataError(f"Fuso IANA non valido: {name}") from exc


def utc_timestamp(value: Any, *, fit_utc: bool = False) -> str:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise DataError("Timestamp ISO 8601 non valido") from exc
    if not isinstance(value, datetime):
        raise DataError("Timestamp mancante")
    if value.tzinfo is None:
        if not fit_utc:
            raise DataError("Timestamp senza offset: usare ISO 8601 con fuso")
        # FIT date_time represents UTC seconds since the FIT epoch, per the protocol.
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


@dataclass(frozen=True)
class Sample:
    timestamp_utc: str
    elapsed_s: float
    heart_rate_bpm: float | None
    speed_m_s: float | None
    distance_m: float | None


@dataclass
class RunData:
    samples: list[Sample]
    source_sha256: str
    timezone: str = "Europe/Rome"
    synthetic: bool = False
    warnings: list[str] = field(default_factory=list)
    pauses_s: list[tuple[float, float]] = field(default_factory=list)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunData:
        if not isinstance(payload, dict):
            raise DataError("JSON normalizzato deve essere un oggetto")
        try:
            if payload["schema_version"] != 1:
                raise DataError("Versione schema non supportata")
            check_timezone(payload["timezone"])
            samples = [
                Sample(**{**row, "timestamp_utc": utc_timestamp(row["timestamp_utc"])})
                for row in payload["samples"]
            ]
            run = cls(
                samples=samples,
                source_sha256=payload["source_sha256"],
                timezone=payload["timezone"],
                synthetic=payload.get("synthetic", False),
                warnings=list(payload.get("warnings", [])),
                pauses_s=[tuple(pair) for pair in payload.get("pauses_s", [])],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataError(f"JSON normalizzato non valido: {exc}") from exc
        validate_run(run)
        return run


def validate_run(run: RunData) -> None:
    check_timezone(run.timezone)
    if not isinstance(run.synthetic, bool) or not all(isinstance(w, str) for w in run.warnings):
        raise DataError("Flag synthetic o avvisi non validi")
    if len(run.samples) < 2:
        raise DataError("Servono almeno due record con timestamp distinti")
    if (
        not isinstance(run.source_sha256, str)
        or len(run.source_sha256) != 64
        or any(c not in "0123456789abcdef" for c in run.source_sha256)
    ):
        raise DataError("SHA-256 sorgente non valido")
    previous = -1.0
    origin = datetime.fromisoformat(utc_timestamp(run.samples[0].timestamp_utc))
    previous_distance = -1.0
    for sample in run.samples:
        timestamp = datetime.fromisoformat(utc_timestamp(sample.timestamp_utc))
        elapsed = number(sample.elapsed_s)
        if (
            not isinstance(sample.elapsed_s, (int, float))
            or isinstance(sample.elapsed_s, bool)
            or elapsed is None
            or elapsed <= previous
        ):
            raise DataError("Tempi trascorsi non strettamente crescenti")
        if abs((timestamp - origin).total_seconds() - elapsed) > 0.001:
            raise DataError("Timestamp e elapsed_s incoerenti")
        previous = elapsed
        for value, maximum in [
            (sample.heart_rate_bpm, 254),
            (sample.speed_m_s, 30),
            (sample.distance_m, math.inf),
        ]:
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or number(value, maximum=maximum) is None
            ):
                raise DataError("Misura fuori dai limiti tecnici o non finita")
        if sample.heart_rate_bpm is not None and sample.heart_rate_bpm == 0:
            raise DataError("FC zero non valida; usare null")
        if sample.distance_m is not None:
            if sample.distance_m < previous_distance:
                raise DataError("Distanza cumulativa decrescente")
            previous_distance = sample.distance_m
    for pair in run.pauses_s:
        if len(pair) != 2:
            raise DataError("Intervallo di pausa deve contenere due valori")
        start, end = pair
        if any(
            not isinstance(value, (float, int)) or isinstance(value, bool) or number(value) is None
            for value in pair
        ):
            raise DataError("Intervallo di pausa non numerico")
        if not (0 <= start < end <= run.samples[-1].elapsed_s):
            raise DataError("Intervallo di pausa non valido")
