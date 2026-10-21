"""Deterministic invented FIT bytes: no downloaded activity or real athlete data."""

from __future__ import annotations

import math
import struct
from datetime import UTC, datetime, timedelta

from .models import Sample

FIT_EPOCH = datetime(1989, 12, 31, tzinfo=UTC)


def crc16(data: bytes) -> int:
    """FIT CRC polynomial, reflected, zero initial value."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    return crc


def _definition(local: int, message: int, fields: list[tuple[int, int, int]]) -> bytes:
    return (
        bytes([0x40 | local, 0, 0])
        + struct.pack("<H", message)
        + bytes([len(fields)])
        + b"".join(bytes(field) for field in fields)
    )


def fit_bytes(
    samples: list[Sample], *, enhanced_speed: bool = False, signals: list[dict] | None = None
) -> bytes:
    if signals is not None and len(signals) != len(samples):
        raise ValueError("Synthetic signal rows must align with samples")
    data = bytearray(_definition(0, 0, [(0, 1, 0), (1, 2, 0x84)]))
    data.extend(struct.pack("<BBH", 0, 4, 255))  # activity; development manufacturer
    data.extend(_definition(1, 18, [(5, 1, 0)]))
    data.extend(bytes([1, 1]))  # session, sport=running
    fields = [(253, 4, 0x86), (3, 1, 2), (5, 4, 0x86), (6, 2, 0x84)]
    if enhanced_speed:
        fields.append((73, 4, 0x86))
    if signals is not None:
        fields.extend([(4, 1, 2), (7, 2, 0x84)])
    data.extend(_definition(2, 20, fields))
    for index, sample in enumerate(samples):
        stamp = datetime.fromisoformat(sample.timestamp_utc)
        seconds = int((stamp - FIT_EPOCH).total_seconds())
        hr = 255 if sample.heart_rate_bpm is None else round(sample.heart_rate_bpm)
        distance = 0xFFFFFFFF if sample.distance_m is None else round(sample.distance_m * 100)
        speed = 0xFFFF if sample.speed_m_s is None else round(sample.speed_m_s * 1000)
        data.extend(struct.pack("<BIBIH", 2, seconds, hr, distance, speed))
        if enhanced_speed:
            data.extend(struct.pack("<I", 0xFFFFFFFF if sample.speed_m_s is None else speed))
        if signals is not None:
            row = signals[index]
            cadence = row.get("cadence_fit_rpm")
            power = row.get("power_w")
            data.extend(
                struct.pack(
                    "<BH",
                    255 if cadence is None else round(cadence),
                    65535 if power is None else round(power),
                )
            )
    header = struct.pack("<BBHI4s", 14, 0x20, 2180, len(data), b".FIT")
    header += struct.pack("<H", crc16(header))
    content = header + bytes(data)
    return content + struct.pack("<H", crc16(content))


def demo_samples() -> list[Sample]:
    start = datetime(2025, 1, 15, 17, 0, tzinfo=UTC)
    return [
        Sample(
            (start + timedelta(seconds=t)).isoformat(),
            float(t),
            None if t in {500, 1000} else round(130 + 12 * t / 1800 + 2 * math.sin(t / 60)),
            3.0,
            3.0 * t,
        )
        for t in range(0, 1801, 5)
    ]
