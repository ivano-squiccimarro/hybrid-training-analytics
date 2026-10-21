"""Explicit proportional trace estimates; never mutate source records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from math import isfinite

from .models import DataError, RunData, number, utc_timestamp, validate_run

GAP_MODEL = "minetti_2002_running_cost_v1"
GAP_SOURCE = "https://doi.org/10.1152/japplphysiol.01177.2001"


@dataclass(frozen=True)
class CalibrationConfig:
    target_distance_m: float
    distance_source: str
    incline_pct: float | None = None
    max_gap_s: float = 10

    def validate(self) -> None:
        for value in (self.target_distance_m, self.max_gap_s):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or number(value) is None
                or value <= 0
            ):
                raise DataError("Distanza dichiarata e gap massimo devono essere finiti e positivi")
        if not isinstance(self.distance_source, str) or self.distance_source not in {
            "watch_final",
            "treadmill_log",
            "user_other",
        }:
            raise DataError("Fonte della distanza dichiarata non supportata")
        if self.incline_pct is not None and (
            not isinstance(self.incline_pct, (int, float))
            or isinstance(self.incline_pct, bool)
            or number(self.incline_pct, minimum=-20, maximum=20) is None
        ):
            raise DataError("Pendenza dichiarata deve essere finita e tra -20 e 20 percento")


def running_cost_j_kg_m(incline_pct: float) -> float:
    """Minetti running cost polynomial; percent converted to dimensionless grade."""
    CalibrationConfig(1, "user_other", incline_pct).validate()
    grade = incline_pct / 100
    return (
        155.4 * grade**5 - 30.4 * grade**4 - 43.3 * grade**3 + 46.3 * grade**2 + 19.5 * grade + 3.6
    )


def calibrate(run: RunData, config: CalibrationConfig) -> dict:
    validate_run(run)
    config.validate()
    first, last = run.samples[0].distance_m, run.samples[-1].distance_m
    if first is None or last is None or last <= first:
        raise DataError("Servono distanze endpoint valide con incremento positivo")
    factor = config.target_distance_m / (last - first)
    if not isfinite(factor) or factor <= 0:
        raise DataError("Fattore di scaling non rappresentabile")
    cost_ratio = (
        None if config.incline_pct is None else running_cost_j_kg_m(config.incline_pct) / 3.6
    )
    samples = []
    previous = None
    for sample in run.samples:
        active = not any(start <= sample.elapsed_s < end for start, end in run.pauses_s)
        distance = None if sample.distance_m is None else (sample.distance_m - first) * factor
        speed = None if sample.speed_m_s is None else sample.speed_m_s * factor
        pace = 1000 / speed if active and speed is not None and speed > 0 else None
        interval_status = "no_previous_record"
        interval_distance = interval_speed = None
        if previous is not None:
            gap = sample.elapsed_s - previous.elapsed_s
            if any(a < sample.elapsed_s and b > previous.elapsed_s for a, b in run.pauses_s):
                interval_status = "timer_pause"
            elif gap > config.max_gap_s:
                interval_status = "gap_exceeds_limit"
            elif previous.distance_m is None or sample.distance_m is None:
                interval_status = "missing_distance"
            else:
                interval_status = "available"
                interval_distance = (sample.distance_m - previous.distance_m) * factor
                interval_speed = interval_distance / gap
        gap_pace = None if pace is None or cost_ratio is None else pace / cost_ratio
        values = (distance, speed, pace, interval_distance, interval_speed, gap_pace)
        if any(value is not None and not isfinite(value) for value in values):
            raise DataError("Stima ricalibrata non rappresentabile; nessun output parziale")
        samples.append(
            {
                "timestamp_utc": utc_timestamp(sample.timestamp_utc),
                "elapsed_s": sample.elapsed_s,
                "heart_rate_bpm": sample.heart_rate_bpm,
                "record_distance_m": sample.distance_m,
                "record_speed_m_s": sample.speed_m_s,
                "scaled_distance_m": distance,
                "scaled_record_speed_m_s": speed,
                "scaled_record_pace_estimate_s_km": pace,
                "scaled_interval_distance_m": interval_distance,
                "scaled_interval_speed_estimate_m_s": interval_speed,
                "interval_status": interval_status,
                "timer_active": active,
                "gap_model_estimate_s_km": gap_pace,
            }
        )
        previous = sample
    warnings = list(run.warnings)
    if run.pauses_s:
        warnings.append("Pause preservate: nessuna stima di intervallo attraverso il timer fermo")
    if any(b.elapsed_s - a.elapsed_s > config.max_gap_s for a, b in pairwise(run.samples)):
        warnings.append(
            "Gap preservati: nessuna distribuzione fra campioni distanti oltre il limite"
        )
    if not run.pauses_s:
        warnings.append("Assenza di pause normalizzate non certifica timer sempre attivo")
    return {
        "schema_version": 1,
        "artifact_type": "proportional_trace_estimate",
        "algorithm_version": "proportional_trace_v1",
        "source_sha256": run.source_sha256,
        "synthetic": run.synthetic,
        "timezone": run.timezone,
        "config": asdict(config),
        "reference_record_distance_delta_m": last - first,
        "source_distance_origin_m": first,
        "scale_factor": factor,
        "pauses_s": list(run.pauses_s),
        "provenance": {
            "heart_rate_bpm": "fit_recorded",
            "record_distance_m": "fit_device_estimate",
            "record_speed_m_s": "fit_device_estimate",
            "target_distance_m": "user_declared_" + config.distance_source,
            "scaled_distance_m": "uniform_error_assumption_estimate",
            "scaled_record_speed_m_s": "device_speed_times_uniform_factor_estimate",
            "scaled_record_pace_estimate_s_km": "inverse_scaled_device_speed_estimate",
            "scaled_interval_speed_estimate_m_s": "adjacent_scaled_distance_over_elapsed_estimate",
            "incline_pct": None
            if config.incline_pct is None
            else "user_declared_constant_treadmill",
            "gap_model_estimate_s_km": None if cost_ratio is None else GAP_MODEL,
        },
        "gap_model": {
            "status": "unavailable" if cost_ratio is None else "available",
            "reason": "Pendenza non dichiarata" if cost_ratio is None else None,
            "name": GAP_MODEL,
            "source": GAP_SOURCE,
            "cost_ratio_to_level": cost_ratio,
            "definition": "scaled_pace * C(0) / C(incline_pct / 100)",
        },
        "warnings": sorted(set(warnings)),
        "limits": [
            "Distanza finale dichiarata; errore uniforme assunto, non dimostrato",
            "Velocità e distanza dispositivo sono canali distinti: gli integrali possono differire",
            "Passo da velocità dispositivo stimata; non misura istantanea indipendente",
            "Null e timer preservati; nessun campione aggiunto, gap interpolato o pausa compressa",
            "GAP di modello a pendenza dichiarata costante; nessuna calibrazione individuale",
            "Range pendenza +/-20 percento è limite tecnico del prodotto, non garanzia di validità",
            "Scaling non conferma automaticamente velocità affidabile o disponibilità della deriva",
        ],
        "samples": samples,
    }
