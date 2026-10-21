import csv

import pytest

from hybrid_training_analytics.models import DataError
from hybrid_training_analytics.strength import COLUMNS, read_strength, summarize_strength


def diary(tmp_path, rows):
    path = tmp_path / "diary.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    return path


def test_zavorra_is_separate_from_bodyweight(tmp_path):
    path = diary(
        tmp_path,
        [["2025-01-01T18:00:00+01:00", "synthetic", "dip", "bodyweight", 80, 10, 0, 1, 6, 2, 8]],
    )
    summary = summarize_strength(read_strength(path))[0]
    assert summary["external_volume_kg_reps"] == 60
    assert summary["system_mass_proxy_kg_reps"] == 540
    assert summary["best_observed_rir"] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("reps", 3.5),
        ("rir", -1),
        ("body_mass_kg", "nan"),
        ("timestamp", "2025-01-01T18:00:00"),
        ("external_load_kg", 20),
        ("rpe", 11),
    ],
)
def test_invalid_diary_rejected(tmp_path, field, value):
    row = dict(
        zip(
            COLUMNS,
            ["2025-01-01T18:00:00+01:00", "synthetic", "dip", "bodyweight", 80, 10, 0, 1, 6, 2, 8],
            strict=True,
        )
    )
    row[field] = value
    with pytest.raises(DataError):
        read_strength(diary(tmp_path, [list(row.values())]))


def test_duplicate_sets_rejected(tmp_path):
    row = ["2025-01-01T18:00:00+01:00", "synthetic", "dip", "bodyweight", 80, 10, 0, 1, 6, 2, 8]
    with pytest.raises(DataError, match="duplicata"):
        read_strength(diary(tmp_path, [row, row]))
