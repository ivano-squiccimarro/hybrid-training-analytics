"""Pure, descriptive quality assessment. No workout reconstruction or trust score."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median

from .exercise_vocabulary import normalize_categories
from .models import DataError


@dataclass(frozen=True)
class QualityConfig:
    max_hr_gap_s: float = 10.0
    max_accel_gap_s: float = 2.0
    timestamp_precision_s: float = 1.0
    min_duration_observations: int = 5
    duration_median_ratio: float = 3.5
    duration_mad_multiplier: float = 6.0

    def __post_init__(self):
        from math import isfinite

        values = (
            self.max_hr_gap_s,
            self.max_accel_gap_s,
            self.timestamp_precision_s,
            self.duration_median_ratio,
            self.duration_mad_multiplier,
        )
        if (
            any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v <= 0
                for v in values
            )
            or self.duration_median_ratio <= 1
            or type(self.min_duration_observations) is not int
            or self.min_duration_observations < 3
        ):
            raise DataError("Configurazione qualità non valida")


def assess_strength_quality(raw: dict, config: QualityConfig | None = None) -> dict:
    config = config or QualityConfig()
    sections = {"available": [], "caution": [], "undetermined": []}

    def finding(section, code, text, provenance):
        sections[section].append({"code": code, "text": text, "provenance": provenance})

    duration = raw["duration_s"]
    origin_inferred = "origin_from_recorded_bounds" in raw["warnings"]
    duration_inferred = "duration_from_recorded_bounds" in raw["warnings"]
    inferred_bounds = origin_inferred or duration_inferred
    time_note = ""
    if origin_inferred:
        time_note += " Origine temporale derivata dagli estremi registrati."
    if duration_inferred:
        time_note += " Durata derivata dagli estremi registrati."
    if not time_note:
        time_note = " Tempo della seduta, non tempo di lavoro muscolare."
    finding(
        "caution" if inferred_bounds else "available",
        "session_time",
        f"Inizio {raw['start_utc']} · durata {duration:g} s." + time_note,
        "inferred" if inferred_bounds else "recorded",
    )
    records = raw["records"]
    valid_hr = sorted(
        (
            r
            for r in records
            if r["heart_rate_bpm"] is not None
            and r["elapsed_s"] is not None
            and 0 <= r["elapsed_s"] <= duration
        ),
        key=lambda r: r["elapsed_s"],
    )
    # Adjacent records only: null samples break support; endpoints outside are excluded.
    ordered = sorted(
        (r for r in records if r["elapsed_s"] is not None and 0 <= r["elapsed_s"] <= duration),
        key=lambda r: r["elapsed_s"],
    )
    gaps = [b["elapsed_s"] - a["elapsed_s"] for a, b in zip(ordered, ordered[1:], strict=False)]
    support = sum(
        b["elapsed_s"] - a["elapsed_s"]
        for a, b in zip(ordered, ordered[1:], strict=False)
        if a["heart_rate_bpm"] is not None
        and b["heart_rate_bpm"] is not None
        and 0 < b["elapsed_s"] - a["elapsed_s"] <= config.max_hr_gap_s
    )
    hr = {
        "records": len(records),
        "valid_in_bounds": len(valid_hr),
        "null_values": sum(r["heart_rate_bpm"] is None for r in records),
        "supported_duration_s": support,
        "temporal_coverage_fraction": support / duration,
        "max_record_gap_s": max(gaps, default=None),
        "large_gap_count": sum(g > config.max_hr_gap_s for g in gaps),
        "provenance": {"values": "recorded", "support": "inferred"},
    }
    if valid_hr:
        finding(
            "available",
            "hr_present",
            f"FC registrata: {len(valid_hr)} campioni in bpm; supporto temporale "
            f"{support:g} s su {duration:g} s ({100 * support / duration:.1f}%), calcolato "
            "dai campioni disponibili.",
            "recorded",
        )
        finding(
            "caution",
            "hr_accuracy_unknown",
            "Copertura non significa accuratezza. Sensore non dichiarato per questa seduta. "
            "La FC non identifica esercizi o confini delle serie.",
            "unavailable",
        )
    else:
        finding("undetermined", "hr_missing", "FC utilizzabile non disponibile.", "unavailable")
    if hr["large_gap_count"] or hr["null_values"] or support < duration:
        finding(
            "caution",
            "hr_support_partial",
            f"FC parziale · valori mancanti: {hr['null_values']}; "
            f"{hr['large_gap_count']} gap oltre {config.max_hr_gap_s:g} s "
            "(limite tecnico, non soglia fisiologica). Nessun riempimento dei campioni.",
            "inferred",
        )
    axes_g = tuple("accel_" + a + "_g" for a in "xyz")
    axes_counts = tuple("accel_" + a + "_counts" for a in "xyz")
    accel = raw["accelerometer"]
    usable = [
        a
        for a in accel
        if a["elapsed_s"] is not None
        and 0 <= a["elapsed_s"] <= duration
        and (all(a[k] is not None for k in axes_g) or all(a[k] is not None for k in axes_counts))
    ]
    accel_times = sorted(set(a["elapsed_s"] for a in usable))
    # No support across incomplete samples or between counts and calibrated axes.
    ordered_accel = sorted(
        (a for a in accel if a["elapsed_s"] is not None and 0 <= a["elapsed_s"] <= duration),
        key=lambda a: a["elapsed_s"],
    )

    def basis(a):
        return (
            "g"
            if all(a[k] is not None for k in axes_g)
            else ("counts" if all(a[k] is not None for k in axes_counts) else None)
        )

    accel_support = sum(
        b["elapsed_s"] - a["elapsed_s"]
        for a, b in zip(ordered_accel, ordered_accel[1:], strict=False)
        if basis(a) is not None
        and basis(a) == basis(b)
        and 0 < b["elapsed_s"] - a["elapsed_s"] <= config.max_accel_gap_s
    )
    accel_stats = {
        "samples": len(accel),
        "supported_duration_s": accel_support,
        "temporal_coverage_fraction": accel_support / duration,
        "large_gap_count": sum(
            b - a > config.max_accel_gap_s
            for a, b in zip(accel_times, accel_times[1:], strict=False)
        ),
        "calibrated_g_samples": sum(basis(a) == "g" for a in usable),
        "counts_samples": sum(basis(a) == "counts" for a in usable),
        "provenance": {"values": "recorded", "support": "inferred"},
        "usable_in_bounds": len(usable),
        "missing_time": sum(a["elapsed_s"] is None for a in accel),
        "max_sample_gap_s": max(
            (b - a for a, b in zip(accel_times, accel_times[1:], strict=False)), default=None
        ),
    }
    if usable:
        finding(
            "available",
            "movement_recorded",
            f"Movimento registrato: {len(usable)} campioni con tempi e tre assi "
            f"in g o conteggi FIT; supporto temporale {accel_support:g} s su {duration:g} s. "
            "Non misura l’attivazione muscolare.",
            "recorded",
        )
    else:
        finding(
            "undetermined",
            "movement_unavailable",
            "Nessun accelerometro utilizzabile decodificato. Non ricostruiamo serie "
            "o muscoli dalla sola FC; messaggi proprietari restano non interpretati.",
            "unavailable",
        )
    if accel_stats["large_gap_count"]:
        finding(
            "caution",
            "movement_gaps",
            f"{accel_stats['large_gap_count']} gap di movimento oltre "
            f"{config.max_accel_gap_s:g} s; supporto parziale, nessun campione inventato.",
            "inferred",
        )
    if len(usable) != len(accel):
        finding(
            "caution",
            "movement_partial",
            f"{len(accel) - len(usable)} campioni di movimento senza assi/tempi utilizzabili "
            "o fuori seduta, preservati nel raw.",
            "inferred",
        )
    sets = raw["device_sets"]
    valid = [
        s
        for s in sets
        if s["start_s"] is not None
        and s["end_s"] is not None
        and 0 <= s["start_s"] < s["end_s"] <= duration
    ]
    invalid = [s["device_set_id"] for s in sets if s not in valid]
    overlaps, holes, precision = [], [], []
    # Sweep maximum end detects nested intervals as well as adjacent overlap.
    end = None
    for row in sorted(valid, key=lambda s: s["start_s"]):
        if end is not None:
            delta = row["start_s"] - end
            issue = {"device_set_id": row["device_set_id"], "delta_s": delta}
            if delta < 0:
                overlaps.append(issue)
            elif delta > 0:
                holes.append(issue)
            if 0 < abs(delta) <= config.timestamp_precision_s:
                precision.append(issue)
        end = max(end if end is not None else row["end_s"], row["end_s"])
    structure = {
        "invalid_set_ids": invalid,
        "overlaps": overlaps,
        "gaps": holes,
        "precision_ambiguities": precision,
    }
    if sets:
        finding(
            "available",
            "device_annotations",
            f"Marcature Garmin presenti: {len(sets)}. Sono indizi del dispositivo, "
            "non serie verificate.",
            "device_hint",
        )
        finding(
            "caution",
            "device_values_unconfirmed",
            "Nomi, ripetizioni e kg Garmin non confermati. Zero non conferma corpo libero "
            "o assenza di lavoro.",
            "device_hint",
        )
    else:
        finding(
            "undetermined",
            "device_annotations_missing",
            "Marcature delle serie non disponibili; non ne inventiamo.",
            "unavailable",
        )
    if invalid or overlaps or holes:
        finding(
            "caution",
            "device_structure",
            f"Marcature con estremi mancanti/non validi/fuori seduta: {len(invalid)}; "
            f"sovrapposizioni: {len(overlaps)}; intervalli scoperti interni: {len(holes)}. "
            "Nessuna correzione automatica.",
            "inferred",
        )
    if precision:
        finding(
            "caution",
            "timestamp_precision",
            f"Discrepanze entro {config.timestamp_precision_s:g} s: {len(precision)}. "
            "Possibili precisioni temporali diverse; causa non dimostrata. "
            "I tempi originali restano invariati.",
            "inferred",
        )
    duration_checks = []
    for kind in ("active", "rest"):
        rows = [
            s
            for s in sets
            if s["device_set_type"] == kind and s["duration_s"] is not None and s["duration_s"] > 0
        ]
        durations = [s["duration_s"] for s in rows]
        enough = len(rows) >= config.min_duration_observations
        middle = median(durations) if enough else None
        mad = median(abs(x - middle) for x in durations) if enough else None
        cutoff = (
            max(
                middle * config.duration_median_ratio,
                middle + config.duration_mad_multiplier * 1.4826 * mad,
            )
            if enough
            else None
        )
        outliers = [s["device_set_id"] for s in rows if enough and s["duration_s"] > cutoff]
        duration_checks.append(
            {
                "device_set_type": kind,
                "observations": len(rows),
                "median_s": middle,
                "mad_s": mad,
                "upper_limit_s": cutoff,
                "long_set_ids": outliers,
                "supported": enough,
            }
        )
        if outliers:
            finding(
                "caution",
                "relative_duration_" + kind,
                f"Blocchi di {'lavoro' if kind == 'active' else 'pausa'} molto più lunghi "
                f"degli altri dello stesso tipo: {len(outliers)}; mediana {middle:g} s, "
                f"criterio superiore {cutoff:g} s. "
                "Confronto interno descrittivo: non dimostra serie errate o dimenticate.",
                "inferred",
            )
        elif not enough:
            finding(
                "undetermined",
                "duration_support_" + kind,
                f"Confronto durate {'lavoro' if kind == 'active' else 'pausa'}: "
                f"osservazioni {len(rows)}, minimo {config.min_duration_observations}. "
                "Nessuna conclusione.",
                "unavailable",
            )
    candidates = [
        {
            "device_set_id": s["device_set_id"],
            "candidates": normalize_categories(s.get("category_device")),
        }
        for s in sets
    ]
    finding(
        "undetermined",
        "true_workout_unknown",
        "Esercizi realmente eseguiti, muscoli coinvolti, serie effettive, RPE/RIR "
        "e carichi muscolari non determinabili dal solo FIT senza dichiarazioni o verifica.",
        "unavailable",
    )
    inventory = raw.get("message_counts")
    ignored = (
        None
        if inventory is None
        else {
            k: v
            for k, v in inventory.items()
            if k not in {"session", "record", "set", "lap", "exercise_title", "accelerometer_data"}
        }
    )
    if inventory is None:
        finding(
            "undetermined",
            "inventory_unavailable",
            "Inventario messaggi assente nel vecchio import; il raw non viene riscritto.",
            "unavailable",
        )
    elif any(k.startswith("unknown") or k == "gyroscope_data" for k in ignored):
        finding(
            "caution",
            "uninterpreted_messages",
            "Messaggi non interpretati presenti (conteggi nell’inventario tecnico). "
            "Non sono prove di un segnale di movimento.",
            "recorded",
        )
    if raw["warnings"]:
        finding(
            "caution",
            "parser_warnings",
            "Il parser segnala limitazioni; codici e dettagli nel riepilogo tecnico.",
            "recorded",
        )
    return {
        "schema_version": 1,
        "artifact_type": "strength_data_quality",
        "source_sha256": raw["source_sha256"],
        "synthetic": raw["synthetic"],
        "config": asdict(config),
        "time_provenance": {
            "start_utc": "inferred" if origin_inferred else "recorded",
            "duration_s": "inferred" if duration_inferred else "recorded",
        },
        "sections": sections,
        "hr": hr,
        "accelerometer": accel_stats,
        "device_structure": structure,
        "duration_checks": duration_checks,
        "native_categories": candidates,
        "message_counts": inventory,
        "uninterpreted_message_counts": ignored,
        "parser_warnings": list(raw["warnings"]),
        "limits": [
            "Nessun voto globale di attendibilità o probabilità di esercizio.",
            "Famiglie di movimento: convenzione del prodotto, non misura muscolare.",
            "Lettura ed esportazione non richiedono correzioni delle serie.",
        ],
    }
