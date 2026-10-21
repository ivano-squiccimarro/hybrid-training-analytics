"""Descriptive metrics and explicitly labelled inference eligibility."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from statistics import median

from .models import DataError, RunData, number, validate_run


@dataclass(frozen=True)
class TreadmillSegment:
    start_s: float
    end_s: float
    speed_kmh: float
    incline_pct: float | None = None


@dataclass(frozen=True)
class AnalysisConfig:
    speed_reliable: bool = False
    constant_incline: bool = False
    warmup_s: float = 300
    min_drift_duration_s: float = 600
    speed_tolerance_fraction: float = 0.03
    max_gap_s: float = 10
    segment_start_s: float | None = None
    segment_end_s: float | None = None
    rpe: float | None = None

    def validate(self) -> None:
        if not isinstance(self.speed_reliable, bool) or not isinstance(self.constant_incline, bool):
            raise DataError("Conferme velocità e pendenza devono essere booleani")
        for value in (
            self.warmup_s,
            self.min_drift_duration_s,
            self.speed_tolerance_fraction,
            self.max_gap_s,
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or number(value) is None
            ):
                raise DataError("Parametri di analisi non finiti o negativi")
        if self.min_drift_duration_s <= 0 or self.max_gap_s <= 0:
            raise DataError("Durata minima e gap massimo devono essere positivi")
        if not 0 < self.speed_tolerance_fraction <= 0.2:
            raise DataError("Tolleranza relativa velocità deve essere in (0, 0.2]")
        for value in (self.segment_start_s, self.segment_end_s):
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or number(value) is None
            ):
                raise DataError("Limiti segmento non validi")
        if self.rpe is not None and (
            not isinstance(self.rpe, (int, float))
            or number(self.rpe, minimum=1, maximum=10) is None
        ):
            raise DataError("RPE deve essere compreso tra 1 e 10")
        if (
            self.segment_start_s is not None
            and self.segment_end_s is not None
            and self.segment_start_s >= self.segment_end_s
        ):
            raise DataError("Inizio segmento deve precedere fine segmento")


def validate_treadmill(segments: list[TreadmillSegment]) -> None:
    for segment in segments:
        for value in (segment.start_s, segment.end_s, segment.speed_kmh):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or number(value) is None
            ):
                raise DataError("Segmento tapis con valori non numerici o non finiti")
        if segment.start_s >= segment.end_s or not 0 < segment.speed_kmh <= 40:
            raise DataError("Intervallo/velocità tapis non validi")
        if segment.incline_pct is not None and (
            not isinstance(segment.incline_pct, (int, float))
            or number(
                segment.incline_pct,
                minimum=-20,
                maximum=40,
            )
            is None
        ):
            raise DataError("Pendenza tapis non valida")
    if any(b.start_s < a.end_s for a, b in pairwise(segments)):
        raise DataError("Segmenti tapis sovrapposti o fuori ordine")


def read_treadmill(path: Path) -> list[TreadmillSegment]:
    result = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {"start_s", "end_s", "speed_kmh", "incline_pct"}
        if set(reader.fieldnames or []) != expected or len(reader.fieldnames or []) != 4:
            raise DataError(f"Colonne tapis roulant attese: {', '.join(sorted(expected))}")
        for index, row in enumerate(reader, 2):
            values = [number(row[key]) for key in ("start_s", "end_s", "speed_kmh")]
            if any(value is None for value in values):
                raise DataError(f"Tapis roulant riga {index}: valori non validi")
            start, end, speed = values
            incline = (
                None
                if not row["incline_pct"]
                else number(
                    row["incline_pct"],
                    minimum=-20,
                    maximum=40,
                )
            )
            if row["incline_pct"] and incline is None:
                raise DataError(f"Tapis roulant riga {index}: pendenza non valida")
            if start >= end or speed <= 0 or speed > 40:
                raise DataError(f"Tapis roulant riga {index}: intervallo/velocità non validi")
            result.append(TreadmillSegment(start, end, speed, incline))
    result.sort(key=lambda segment: segment.start_s)
    if not result:
        raise DataError("CSV tapis roulant vuoto")
    validate_treadmill(result)
    return result


def _paused(run: RunData, start: float, end: float) -> bool:
    return any(a < end and b > start for a, b in run.pauses_s)


def weighted_mean(
    run: RunData,
    field: str,
    start: float,
    end: float,
    max_gap_s: float,
) -> tuple[float | None, float]:
    """Trapezoids within valid adjacent pairs only; never bridge missing data or pauses."""
    integral = duration = 0.0
    for a, b in pairwise(run.samples):
        left, right = max(start, a.elapsed_s), min(end, b.elapsed_s)
        gap = b.elapsed_s - a.elapsed_s
        va, vb = getattr(a, field), getattr(b, field)
        if (
            right <= left
            or gap > max_gap_s
            or va is None
            or vb is None
            or _paused(run, a.elapsed_s, b.elapsed_s)
        ):
            continue
        vl = va + (vb - va) * (left - a.elapsed_s) / gap
        vr = va + (vb - va) * (right - a.elapsed_s) / gap
        integral += (vl + vr) / 2 * (right - left)
        duration += right - left
    return (integral / duration if duration else None, duration)


def analyze(
    run: RunData,
    config: AnalysisConfig | None = None,
    treadmill: list[TreadmillSegment] | None = None,
) -> dict:
    validate_run(run)
    config = config or AnalysisConfig()
    config.validate()
    treadmill = treadmill or []
    validate_treadmill(treadmill)
    duration = run.samples[-1].elapsed_s
    if any(segment.end_s > duration for segment in treadmill):
        raise DataError("Segmento tapis roulant oltre la durata registrata")
    if config.segment_end_s is not None and config.segment_end_s > duration:
        raise DataError("Fine segmento oltre la durata registrata")
    times = [s.elapsed_s for s in run.samples]
    gaps = [b - a for a, b in pairwise(times)]
    jumps = [
        b.elapsed_s
        for a, b in pairwise(run.samples)
        if a.heart_rate_bpm is not None
        and b.heart_rate_bpm is not None
        and b.elapsed_s - a.elapsed_s <= 5
        and abs(b.heart_rate_bpm - a.heart_rate_bpm) >= 20
    ]
    frozen_duration = frozen_max = 0.0
    for a, b in pairwise(run.samples):
        if (
            a.heart_rate_bpm is not None
            and a.heart_rate_bpm == b.heart_rate_bpm
            and b.elapsed_s - a.elapsed_s <= config.max_gap_s
            and not _paused(run, a.elapsed_s, b.elapsed_s)
        ):
            frozen_duration += b.elapsed_s - a.elapsed_s
            frozen_max = max(frozen_max, frozen_duration)
        else:
            frozen_duration = 0.0
    hr_mean, hr_duration = weighted_mean(
        run,
        "heart_rate_bpm",
        0,
        duration,
        config.max_gap_s,
    )
    warnings = list(run.warnings)
    if jumps:
        warnings.append("Salti FC >=20 bpm in <=5 s: controllo euristico, segnale sospetto")
    if frozen_max >= 120:
        warnings.append("FC identica per >=120 s: possibile segnale congelato; verificare sensore")
    if max(gaps) > config.max_gap_s:
        warnings.append("Gap temporali oltre il limite: nessuna interpolazione attraverso i gap")
    if not config.speed_reliable:
        warnings.append("Passo Garmin non confermato: nessun passo calibrato pubblicato")
    distances = [s.distance_m for s in run.samples if s.distance_m is not None]
    distance = distances[-1] - distances[0] if len(distances) >= 2 else None
    comparison: dict = {"status": "unavailable", "reason": "Nessun log tapis roulant completo"}
    if treadmill:
        complete = (
            treadmill[0].start_s == 0
            and treadmill[-1].end_s == duration
            and all(a.end_s == b.start_s for a, b in pairwise(treadmill))
        )
        treadmill_distance = sum(
            (segment.end_s - segment.start_s) * segment.speed_kmh / 3.6 for segment in treadmill
        )
        if (
            complete
            and run.samples[0].distance_m is not None
            and (run.samples[-1].distance_m is not None and distance is not None)
        ):
            comparison = {
                "status": "available",
                "kind": "estimate",
                "treadmill_setting_distance_m": treadmill_distance,
                "garmin_record_distance_m": distance,
                "difference_m": distance - treadmill_distance,
                "difference_pct": 100 * (distance - treadmill_distance) / treadmill_distance,
                "note": "Stesso intervallo; impostazione tapis non certificata. "
                "Nessuna correzione.",
            }
        else:
            comparison["reason"] = "Intervalli non coincidenti o distanze endpoint mancanti"
    start = max(config.warmup_s, config.segment_start_s or 0)
    end = config.segment_end_s if config.segment_end_s is not None else duration
    reference = "garmin_user_confirmed"
    candidate = None
    if treadmill:
        candidates = [
            (max(start, s.start_s), min(end, s.end_s), s)
            for s in treadmill
            if min(end, s.end_s) > max(start, s.start_s)
        ]
        if candidates:
            start, end, candidate = max(candidates, key=lambda item: item[1] - item[0])
            reference = "treadmill_setting_user_log"
    selected = [s for s in run.samples if start <= s.elapsed_s <= end]
    speeds = [s.speed_m_s for s in selected if s.speed_m_s is not None]
    reasons = []
    if end - start < config.min_drift_duration_s:
        reasons.append("Finestra troppo breve dopo il riscaldamento")
    if not selected or len(selected) < 2:
        reasons.append("Campioni insufficienti nella finestra")
    if _paused(run, start, end):
        reasons.append("Pausa timer nella finestra")
    if any(start <= jump <= end for jump in jumps):
        reasons.append("Salti FC sospetti nella finestra")
    if frozen_max >= 120:
        reasons.append("Possibile FC congelata; verifica manuale necessaria")
    if any(a < end and b > start and b - a > config.max_gap_s for a, b in pairwise(times)):
        reasons.append("Gap temporale nella finestra")
    if candidate is None:
        if not config.speed_reliable:
            reasons.append("Velocità non confermata dall'utente")
        if not config.constant_incline:
            reasons.append("Pendenza costante non confermata dall'utente")
        if not speeds or len(speeds) < len(selected) or min(speeds) <= 0:
            reasons.append("Velocità mancanti o nulle nella finestra")
        elif max(abs(s / median(speeds) - 1) for s in speeds) > config.speed_tolerance_fraction:
            reasons.append("Velocità non costante entro la tolleranza tecnica")
    elif candidate.incline_pct is None:
        reasons.append("Pendenza tapis non dichiarata costante nel log")
    midpoint = (start + end) / 2
    first, covered_first = weighted_mean(run, "heart_rate_bpm", start, midpoint, config.max_gap_s)
    second, covered_second = weighted_mean(run, "heart_rate_bpm", midpoint, end, config.max_gap_s)
    half = (end - start) / 2
    if half <= 0 or min(covered_first, covered_second) < 0.9 * half:
        reasons.append("Copertura FC valida <90% in una metà")
    drift = {
        "status": "unavailable" if reasons else "available",
        "kind": "inference",
        "reasons": reasons,
        "start_s": start,
        "end_s": end,
        "reference": reference,
        "definition": "100 * (FC media seconda metà / FC media prima metà - 1)",
        "heart_rate_change_pct": None
        if reasons or not first or second is None
        else 100 * (second / first - 1),
        "first_half_bpm": first,
        "second_half_bpm": second,
        "note": "Variazione descrittiva FC a carico dichiarato costante; non soglia o diagnosi.",
    }
    speed_mean, speed_duration = weighted_mean(run, "speed_m_s", 0, duration, config.max_gap_s)
    pace = None
    if config.speed_reliable and speed_mean and speed_duration >= 0.9 * duration:
        pace = 1000 / speed_mean
    return {
        "schema_version": 1,
        "synthetic": run.synthetic,
        "source_sha256": run.source_sha256,
        "timezone": run.timezone,
        "start_utc": run.samples[0].timestamp_utc,
        "config": asdict(config),
        "treadmill_segments": [asdict(s) for s in treadmill],
        "observations": {
            "records": len(run.samples),
            "elapsed_duration_s": duration,
            "garmin_record_distance_delta_m": distance,
            "heart_rate_time_weighted_bpm": hr_mean,
            "rpe_self_report_1_10": config.rpe,
        },
        "estimates": {"pace_s_km_user_confirmed_speed": pace},
        "quality": {
            "hr_time_coverage_fraction": hr_duration / duration,
            "missing_hr_records": sum(s.heart_rate_bpm is None for s in run.samples),
            "missing_speed_records": sum(s.speed_m_s is None for s in run.samples),
            "median_record_interval_s": median(gaps),
            "max_record_gap_s": max(gaps),
            "suspicious_hr_jump_count": len(jumps),
            "longest_constant_hr_s": frozen_max,
            "sensor_accuracy": "unknown",
        },
        "drift": drift,
        "comparison": comparison,
        "warnings": sorted(set(warnings)),
        "limits": [
            "Nessuna soglia FC, zona, VO2max o prescrizione automatica",
            "La copertura del segnale non dimostra accuratezza",
            "Velocità/distanza indoor Garmin restano stime del dispositivo",
        ],
    }
