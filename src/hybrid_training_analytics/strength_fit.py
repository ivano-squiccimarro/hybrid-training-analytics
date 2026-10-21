"""Strength FIT whitelist: recorded channels and device annotations stay separate."""

from __future__ import annotations

import hashlib
import io
from datetime import datetime, timedelta
from pathlib import Path

import fitdecode

from .models import DataError, number, utc_timestamp


def _array(value):
    return list(value) if isinstance(value, (list, tuple)) else [] if value is None else [value]


def _finite(value, lo=-float("inf"), hi=float("inf")):
    return number(value, minimum=lo, maximum=hi)


def read_strength_fit(path: Path, *, synthetic: bool = False) -> dict:
    if not 0 < path.stat().st_size <= 10 * 1024 * 1024:
        raise DataError("FIT vuoto o oltre 10 MiB")
    content = path.read_bytes()
    records = []
    sets = []
    laps = []
    accel = []
    titles = []
    sessions = []
    warnings = []
    headers = 0
    message_counts = {}
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

                message_counts[frame.name] = message_counts.get(frame.name, 0) + 1

                def get(name, frame=frame):
                    return frame.get_value(name, fallback=None)

                if frame.name == "session":
                    sessions.append(
                        {
                            key: get(key)
                            for key in (
                                "sport",
                                "sub_sport",
                                "start_time",
                                "timestamp",
                                "total_elapsed_time",
                            )
                        }
                    )
                elif frame.name == "record":
                    stamp = get("timestamp")
                    if stamp is None:
                        warnings.append("hr_record_without_timestamp")
                        continue
                    records.append(
                        {
                            "timestamp_utc": utc_timestamp(stamp, fit_utc=True),
                            "heart_rate_bpm": number(get("heart_rate"), minimum=1, maximum=254),
                        }
                    )
                elif frame.name == "set":
                    start = get("start_time")
                    duration = number(get("duration"))
                    sets.append(
                        {
                            "device_set_id": f"fit-set-{len(sets)}",
                            "start_utc": None
                            if start is None
                            else utc_timestamp(start, fit_utc=True),
                            "timestamp_utc": None
                            if get("timestamp") is None
                            else utc_timestamp(get("timestamp"), fit_utc=True),
                            "duration_s": duration,
                            "device_set_type": get("set_type"),
                            "repetitions_device": number(get("repetitions"), maximum=65534),
                            "weight_device_kg": number(get("weight")),
                            "category_device": get("category"),
                            "subtype_device": get("category_subtype"),
                            "step_index_device": get("wkt_step_index"),
                        }
                    )
                elif frame.name == "lap":
                    start, end = get("start_time"), get("timestamp")
                    laps.append(
                        {
                            "start_utc": None
                            if start is None
                            else utc_timestamp(start, fit_utc=True),
                            "end_utc": None if end is None else utc_timestamp(end, fit_utc=True),
                            "duration_device_s": number(get("total_elapsed_time")),
                        }
                    )
                elif frame.name == "exercise_title":
                    titles.append(
                        {
                            key: get(key)
                            for key in (
                                "exercise_category",
                                "exercise_name",
                                "wkt_step_name",
                                "message_index",
                            )
                        }
                    )
                elif frame.name == "accelerometer_data":
                    stamp = get("timestamp")
                    millis = number(get("timestamp_ms"), maximum=999)
                    offsets = _array(get("sample_time_offset"))
                    raw_axes = [_array(get("accel_" + axis)) for axis in "xyz"]
                    calibrated = [_array(get("calibrated_accel_" + axis)) for axis in "xyz"]
                    compressed = [
                        _array(get("compressed_calibrated_accel_" + axis)) for axis in "xyz"
                    ]
                    arrays = [offsets, *raw_axes, *calibrated, *compressed]
                    count = max((len(a) for a in arrays), default=0)
                    if count > 30:
                        raise DataError("Pacchetto accelerometro oltre 30 campioni non supportato")
                    if any(a and len(a) != count for a in arrays):
                        warnings.append("accelerometer_array_length_mismatch")
                    for index in range(count):

                        def at(a, index=index):
                            return a[index] if index < len(a) else None

                        offset = number(at(offsets))
                        timestamp = None
                        if stamp is not None and millis is not None and offset is not None:
                            timestamp = (
                                datetime.fromisoformat(utc_timestamp(stamp, fit_utc=True))
                                + timedelta(milliseconds=millis + offset)
                            ).isoformat()
                        else:
                            warnings.append("accelerometer_timestamp_incomplete")
                        g = []
                        basis = []
                        for direct, packed in zip(calibrated, compressed, strict=True):
                            value = _finite(at(direct))
                            if value is not None:
                                g.append(value)
                                basis.append("FIT calibrated g")
                            else:
                                value = _finite(at(packed))
                                g.append(None if value is None else value / 1000)
                                basis.append(None if value is None else "FIT compressed mG / 1000")
                        accel.append(
                            {
                                "timestamp_utc": timestamp,
                                "accel_x_g": g[0],
                                "accel_y_g": g[1],
                                "accel_z_g": g[2],
                                "accel_x_counts": number(at(raw_axes[0]), maximum=65534),
                                "accel_y_counts": number(at(raw_axes[1]), maximum=65534),
                                "accel_z_counts": number(at(raw_axes[2]), maximum=65534),
                                "axis_provenance": basis,
                            }
                        )
    except (fitdecode.FitError, ValueError, KeyError, EOFError) as exc:
        raise DataError(f"FIT forza non leggibile o CRC non valido: {exc}") from exc
    if headers != 1 or len(sessions) != 1:
        raise DataError("Richiesta singola sessione FIT; concatenati/multisport non supportati")
    session = sessions[0]
    if session["sub_sport"] != "strength_training" and not (
        session["sport"] == "fitness_equipment" and sets
    ):
        raise DataError("FIT non identificato come allenamento forza")
    if session["sub_sport"] != "strength_training":
        warnings.append("strength_subtype_not_declared_device_sets_present")
    timestamps = [r["timestamp_utc"] for r in records] + [
        a["timestamp_utc"] for a in accel if a["timestamp_utc"]
    ]
    timestamps += [s["start_utc"] for s in sets if s["start_utc"]]
    origin = (
        utc_timestamp(session["start_time"], fit_utc=True)
        if session["start_time"] is not None
        else min(timestamps, default=None)
    )
    if session["start_time"] is None:
        warnings.append("origin_from_recorded_bounds")
    if origin is None:
        raise DataError("FIT forza privo di origine temporale utilizzabile")
    start = datetime.fromisoformat(origin)
    duration = number(session["total_elapsed_time"])
    if duration is None:
        ends = timestamps + [
            (
                datetime.fromisoformat(s["start_utc"]) + timedelta(seconds=s["duration_s"])
            ).isoformat()
            for s in sets
            if s["start_utc"] is not None and s["duration_s"] is not None
        ]
        duration = (datetime.fromisoformat(max(ends, default=origin)) - start).total_seconds()
        warnings.append("duration_from_recorded_bounds")
    if not 0 < duration <= 86400:
        raise DataError("Durata forza mancante o oltre limite tecnico di 24 ore")
    indexed = {}
    previous = ""
    for record in records:
        stamp = record["timestamp_utc"]
        if stamp in indexed:
            warnings.append("duplicate_hr_timestamp_last_record_retained")
        if stamp < previous:
            warnings.append("hr_records_reordered_utc")
        previous = stamp
        indexed[stamp] = record
    records = sorted(indexed.values(), key=lambda r: r["timestamp_utc"])
    for row in records + accel:
        row["elapsed_s"] = (
            None
            if row["timestamp_utc"] is None
            else (datetime.fromisoformat(row["timestamp_utc"]) - start).total_seconds()
        )
        if row["elapsed_s"] is not None and not 0 <= row["elapsed_s"] <= duration:
            warnings.append("samples_outside_session_bounds_retained_not_analyzed")
    for row in sets:
        row["start_s"] = (
            None
            if row["start_utc"] is None
            else (datetime.fromisoformat(row["start_utc"]) - start).total_seconds()
        )
        row["end_s"] = (
            None
            if row["start_s"] is None or row["duration_s"] is None
            else row["start_s"] + row["duration_s"]
        )
        if (
            row["start_s"] is None
            or row["end_s"] is None
            or not 0 <= row["start_s"] < row["end_s"] <= duration
        ):
            warnings.append("device_set_bounds_missing_or_invalid")
    return {
        "schema_version": 1,
        "artifact_type": "strength_fit_raw",
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "synthetic": synthetic,
        "start_utc": origin,
        "duration_s": duration,
        "message_counts": message_counts,
        "records": records,
        "accelerometer": accel,
        "device_sets": sets,
        "device_laps": laps,
        "device_titles": titles,
        "warnings": sorted(set(warnings)),
        "provenance": {
            "records": "FIT recorded HR; source accuracy unknown",
            "accelerometer": (
                "FIT recorded counts / decoded calibrated g; no inferred ADC calibration"
            ),
            "device_sets": "unverified device/user set annotations, not confirmed truth",
        },
        "capabilities": {
            "hr": any(r["heart_rate_bpm"] is not None for r in records),
            "accelerometer": bool(accel),
            "device_sets": bool(sets),
        },
        "limits": [
            "Sensor motion is not muscle activation; HR cannot locate set boundaries reliably",
            "No GPS, serial, profiles or developer fields retained; no samples or loads invented",
        ],
    }
