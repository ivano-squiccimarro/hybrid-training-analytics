"""Invented strength FIT generator for tests/demo; no personal data."""

from __future__ import annotations

import math
import struct
from datetime import UTC, datetime

from .synthetic import FIT_EPOCH, _definition, crc16


def strength_fit_bytes(
    *,
    acceleration=True,
    calibrated=True,
    device_sets=True,
    missing_offsets=False,
    category_codes=None,
    noisy=False,
    missing_start=False,
    missing_duration=False,
) -> bytes:
    start = int((datetime(2025, 2, 1, 12, tzinfo=UTC) - FIT_EPOCH).total_seconds())
    data = bytearray(_definition(0, 0, [(0, 1, 0), (1, 2, 0x84)]))
    data.extend(struct.pack("<BBH", 0, 4, 255))
    data.extend(_definition(1, 18, [(5, 1, 0), (6, 1, 0), (2, 4, 0x86), (7, 4, 0x86)]))
    data.extend(
        struct.pack(
            "<BBBII",
            1,
            4,
            20,
            0xFFFFFFFF if missing_start else start,
            0xFFFFFFFF if missing_duration else 300000,
        )
    )
    data.extend(_definition(2, 20, [(253, 4, 0x86), (3, 1, 2)]))
    for t in range(0, 301, 5):
        if noisy and 60 <= t < 85:
            continue
        hr = 255 if noisy and t == 100 else 130 + round(5 * math.sin(t / 15))
        data.extend(struct.pack("<BIB", 2, start + t, hr))
    if device_sets:
        data.extend(
            _definition(
                3,
                225,
                [(6, 4, 0x86), (0, 4, 0x86), (5, 1, 2), (3, 2, 0x84), (4, 2, 0x84)]
                + ([(7, 2 * len(category_codes), 0x84)] if category_codes else []),
            )
        )
        # Synthetic noisy variant: precision discrepancy, gap, relative long block, outside rest.
        annotations = (
            [(30, 30, 1, 10, 50 * 16)]
            if not noisy
            else [
                (0, 20.4, 1, 8, 0),
                (20, 20, 1, 8, 0),
                (42, 20, 1, 0, 0),
                (64, 20, 1, 8, 0),
                (86, 20, 1, 8, 0),
                (110, 180, 1, 8, 0),
                (295, 20, 0, 0, 0),
            ]
        )
        for begin, duration, kind, reps, weight in annotations:
            data.extend(
                struct.pack("<BIIBHH", 3, start + begin, round(duration * 1000), kind, reps, weight)
            )
            if category_codes:
                data.extend(struct.pack("<" + "H" * len(category_codes), *category_codes))
    if acceleration:
        axis_fields = (
            [(5, 40, 0x88), (6, 40, 0x88), (7, 40, 0x88)]
            if calibrated
            else [(2, 20, 0x84), (3, 20, 0x84), (4, 20, 0x84)]
        )
        fields = (
            [(253, 4, 0x86), (0, 2, 0x84)]
            + ([] if missing_offsets else [(1, 20, 0x84)])
            + axis_fields
        )
        data.extend(_definition(4, 165, fields))
        for t in range(300):
            data.extend(struct.pack("<BIH", 4, start + t, 0))
            if not missing_offsets:
                data.extend(struct.pack("<10H", *(i * 100 for i in range(10))))
            moving = 30 <= t < 60 or 180 <= t < 210
            x = [0.4 * math.sin(i * 2 * math.pi / 10) if moving else 0.0 for i in range(10)]
            axes = [x, [0.0] * 10, [1.0] * 10]
            for values in axes:
                if calibrated:
                    data.extend(struct.pack("<10f", *values))
                else:
                    data.extend(struct.pack("<10H", *(round(32768 + v * 1000) for v in values)))
    header = struct.pack("<BBHI4s", 14, 0x20, 2180, len(data), b".FIT")
    header += struct.pack("<H", crc16(header))
    content = header + bytes(data)
    return content + struct.pack("<H", crc16(content))
