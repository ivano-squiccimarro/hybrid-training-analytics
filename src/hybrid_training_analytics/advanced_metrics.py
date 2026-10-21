"""Temporal support, minute bins, joint-support decoupling and descriptive comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from statistics import median

from .models import DataError, RunData, number, validate_run
from .signals import validate_signals

FIELDS = ("heart_rate_bpm", "speed_m_s", "cadence_fit_rpm", "power_w")


def channels(run: RunData, signals: dict | None = None, factor: float = 1) -> list[dict]:
    validate_run(run)
    if not isinstance(factor, (int, float)) or number(factor, minimum=0.000001) is None:
        raise DataError("Fattore deve essere finito e positivo")
    extras = validate_signals(run, signals)
    return [
        {
            "elapsed_s": s.elapsed_s,
            "heart_rate_bpm": s.heart_rate_bpm,
            "speed_m_s": None if s.speed_m_s is None else s.speed_m_s * factor,
            "cadence_fit_rpm": e["cadence_fit_rpm"],
            "power_w": e["power_w"],
        }
        for s, e in zip(run.samples, extras, strict=True)
    ]


def support(
    run: RunData,
    rows: list[dict],
    fields: tuple[str, ...],
    start: float,
    end: float,
    max_gap_s: float = 10,
    *,
    positive: bool = False,
) -> dict:
    totals = dict.fromkeys(fields, 0.0)
    duration = 0.0
    for a, b in pairwise(rows):
        dt = b["elapsed_s"] - a["elapsed_s"]
        left, right = max(start, a["elapsed_s"]), min(end, b["elapsed_s"])
        if (
            left >= right
            or dt > max_gap_s
            or any(x < b["elapsed_s"] and y > a["elapsed_s"] for x, y in run.pauses_s)
        ):
            continue
        if any(
            a[k] is None or b[k] is None or (positive and (a[k] <= 0 or b[k] <= 0)) for k in fields
        ):
            continue
        span = right - left
        for key in fields:
            lv = a[key] + (b[key] - a[key]) * (left - a["elapsed_s"]) / dt
            rv = a[key] + (b[key] - a[key]) * (right - a["elapsed_s"]) / dt
            totals[key] += (lv + rv) * 0.5 * span
        duration += span
    return {
        "valid_s": duration,
        "coverage_fraction": duration / (end - start),
        "means": {k: v / duration if duration else None for k, v in totals.items()},
        "integrals": {k: v if duration else None for k, v in totals.items()},
    }


def minute_bins(
    run: RunData, signals: dict | None = None, factor: float = 1, max_gap_s: float = 10
) -> list[dict]:
    if number(max_gap_s, minimum=0.001) is None:
        raise DataError("Gap massimo deve essere positivo")
    rows = channels(run, signals, factor)
    result = []
    origin = datetime.fromisoformat(run.samples[0].timestamp_utc)
    start = 0.0
    end_s = run.samples[-1].elapsed_s
    while start < end_s:
        end = min(start + 60, end_s)
        row = {
            "start_utc": (origin + timedelta(seconds=start)).isoformat(),
            "start_s": start,
            "end_s": end,
            "span_s": end - start,
            "speed_scale_factor": factor,
        }
        for field in FIELDS:
            values = support(run, rows, (field,), start, end, max_gap_s)
            row[field] = values["means"][field]
            row[field + "_valid_s"] = values["valid_s"]
        result.append(row)
        start = end
    return result


@dataclass(frozen=True)
class DecouplingConfig:
    start_s: float = 300
    end_s: float | None = None
    output: str = "speed_m_s"
    output_confirmed: bool = False
    constant_incline: bool = False
    max_gap_s: float = 10
    min_duration_s: float = 600
    min_coverage: float = 0.9
    output_tolerance: float = 0.03
    cadence_tolerance: float = 0.1

    def validate(self, run: RunData) -> float:
        end = run.samples[-1].elapsed_s if self.end_s is None else self.end_s
        for value in (
            self.start_s,
            end,
            self.max_gap_s,
            self.min_duration_s,
            self.min_coverage,
            self.output_tolerance,
            self.cadence_tolerance,
        ):
            if not isinstance(value, (int, float)) or number(value) is None:
                raise DataError("Configurazione decoupling: numero non finito o negativo")
        if not 0 <= self.start_s < end <= run.samples[-1].elapsed_s:
            raise DataError("Finestra decoupling fuori attivitÃƒÂ ")
        if self.max_gap_s <= 0 or self.min_duration_s <= 0 or not 0 < self.min_coverage <= 1:
            raise DataError("Durata/gap/copertura non validi")
        if self.output not in {"speed_m_s", "power_w"}:
            raise DataError("Output decoupling: speed_m_s oppure power_w")
        if not isinstance(self.output_confirmed, bool) or not isinstance(
            self.constant_incline, bool
        ):
            raise DataError("Conferme decoupling devono essere booleane")
        return end


def quality_events(
    run: RunData, rows: list[dict], start: float, end: float, max_gap_s: float
) -> list[dict]:
    events = []
    frozen_start = None
    for a, b in pairwise(rows):
        if b["elapsed_s"] <= start or a["elapsed_s"] >= end:
            continue
        left, right = max(start, a["elapsed_s"]), min(end, b["elapsed_s"])
        dt = b["elapsed_s"] - a["elapsed_s"]
        pause = any(x < b["elapsed_s"] and y > a["elapsed_s"] for x, y in run.pauses_s)
        if dt > max_gap_s:
            events.append({"kind": "gap", "start_s": left, "end_s": right})
        if pause:
            events.append({"kind": "timer_pause", "start_s": left, "end_s": right})
        hr1, hr2 = a["heart_rate_bpm"], b["heart_rate_bpm"]
        if not pause and dt <= 5 and hr1 is not None and hr2 is not None and abs(hr2 - hr1) >= 20:
            events.append({"kind": "suspect_hr_jump", "start_s": left, "end_s": right})
        continuous = not pause and dt <= max_gap_s and hr1 is not None and hr1 == hr2
        if continuous:
            frozen_start = left if frozen_start is None else frozen_start
            if right - frozen_start >= 120:
                events.append({"kind": "constant_hr_120s", "start_s": frozen_start, "end_s": right})
                frozen_start = right
        else:
            frozen_start = None
    return events


def decoupling(
    run: RunData,
    config: DecouplingConfig | None = None,
    signals: dict | None = None,
    factor: float = 1,
) -> dict:
    config = config or DecouplingConfig()
    rows = channels(run, signals, factor)
    end = config.validate(run)
    start, middle = config.start_s, (config.start_s + end) / 2
    fields = ("heart_rate_bpm", config.output)
    halves = [
        support(run, rows, fields, a, b, config.max_gap_s, positive=True)
        for a, b in ((start, middle), (middle, end))
    ]
    reasons = []
    if end - start < config.min_duration_s:
        reasons.append("duration_below_technical_minimum")
    if not config.output_confirmed:
        reasons.append("output_not_confirmed_by_user")
    if not config.constant_incline:
        reasons.append("constant_incline_not_declared")
    if any(h["coverage_fraction"] < config.min_coverage for h in halves):
        reasons.append("insufficient_joint_coverage")
    events = quality_events(run, rows, start, end, config.max_gap_s)
    reasons.extend(sorted({e["kind"] for e in events}))
    selected = [r for r in rows if start <= r["elapsed_s"] <= end]
    for field, tolerance, label in (
        (config.output, config.output_tolerance, "variable_output"),
        ("cadence_fit_rpm", config.cadence_tolerance, "variable_cadence"),
    ):
        values = [r[field] for r in selected if r[field] is not None]
        if values:
            mid = median(values)
            if mid <= 0 or any(abs(v - mid) > tolerance * mid for v in values):
                reasons.append(label)
    ef = [
        h["means"][config.output] / h["means"]["heart_rate_bpm"] if h["valid_s"] else None
        for h in halves
    ]
    value = None if reasons or any(v is None for v in ef) else 100 * (ef[0] - ef[1]) / ef[0]
    cadence = support(run, rows, ("cadence_fit_rpm",), start, end, config.max_gap_s)
    return {
        "cadence_summary": cadence,
        "algorithm": "joint_support_decoupling_v1",
        "status": "available" if value is not None else "unavailable",
        "decoupling_pct": value,
        "reasons": sorted(set(reasons)),
        "config": asdict(config),
        "window_s": [start, end],
        "halves": halves,
        "efficiency_factor": ef,
        "efficiency_units": "(m/s)/bpm" if config.output == "speed_m_s" else "W/bpm",
        "speed_scale_factor": factor,
        "quality_events": events,
        "cadence_status": "not_recorded"
        if not any(r["cadence_fit_rpm"] is not None for r in selected)
        else "recorded_native_fit_rpm",
        "formula": (
            "100 * (EF_first - EF_second) / EF_first; "
            "EF=mean(output)/mean(HR) on joint temporal support"
        ),
        "limits": (
            "Technical eligibility, not statistical certainty; no "
            "physiological cause or individual threshold inferred"
        ),
    }


def sensitivity(
    run: RunData, config: DecouplingConfig, signals: dict | None = None, factor: float = 1
) -> list[dict]:
    end = config.validate(run)
    variants = [(config.start_s, end), (config.start_s + 60, end), (config.start_s, end - 60)]
    result = []
    for start, finish in variants:
        if start >= finish:
            continue
        result.append(
            decoupling(
                run,
                DecouplingConfig(**{**asdict(config), "start_s": start, "end_s": finish}),
                signals,
                factor,
            )
        )
    return result


def compare_phases(a: dict, b: dict, *, context_a: dict, context_b: dict) -> dict:
    required = ("sensor", "incline_pct", "output_basis")
    reasons = [
        f"context_missing_or_different:{k}"
        for k in required
        if context_a.get(k) is None or context_a.get(k) != context_b.get(k)
    ]
    if a["status"] != "available" or b["status"] != "available":
        reasons.append("phase_not_eligible")
    if a["config"]["output"] != b["config"]["output"]:
        reasons.append("different_output_channels")
    duration_a = a["window_s"][1] - a["window_s"][0]
    duration_b = b["window_s"][1] - b["window_s"][0]
    if abs(duration_a - duration_b) > 1:
        reasons.append("different_phase_durations")
    output_a = [h["means"][a["config"]["output"]] for h in a["halves"]]
    output_b = [h["means"][b["config"]["output"]] for h in b["halves"]]
    if all(v is not None for v in output_a + output_b):
        mean_a, mean_b = sum(output_a) / 2, sum(output_b) / 2
        if abs(mean_b / mean_a - 1) > min(
            a["config"]["output_tolerance"], b["config"]["output_tolerance"]
        ):
            reasons.append("different_output_levels")
    cadence_a, cadence_b = a["cadence_summary"], b["cadence_summary"]
    cadence_delta = None
    if not reasons and all(c["coverage_fraction"] >= 0.9 for c in (cadence_a, cadence_b)):
        ca, cb = cadence_a["means"]["cadence_fit_rpm"], cadence_b["means"]["cadence_fit_rpm"]
        cadence_delta = 100 * (cb / ca - 1) if ca and cb is not None else None
    return {
        "delta_cadence_pct": cadence_delta,
        "cadence_limit": "Native FIT rpm; delta unavailable without >=90% temporal coverage",
        "status": "unavailable" if reasons else "available",
        "reasons": reasons,
        "delta_decoupling_percentage_points": None
        if reasons
        else b["decoupling_pct"] - a["decoupling_pct"],
        "baseline": a,
        "comparison": b,
        "context_baseline": context_a,
        "context_comparison": context_b,
        "interpretation": (
            "Descriptive matched-phase difference; no causal inference or significance test"
        ),
    }
