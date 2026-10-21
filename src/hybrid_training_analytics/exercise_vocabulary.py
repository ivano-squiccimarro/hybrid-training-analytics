"""Device category candidates, never confirmed exercises or measured muscle use."""

from __future__ import annotations

import fitdecode
from fitdecode.profile import FIELD_TYPES

VOCABULARY_VERSION = 1
# Product convention: movement families overlap, not physiological classifications.
VOCABULARY = {
    "bench_press": ("Panca / spinta orizzontale", ["panca", "bench_press"], ["spinta"]),
    "curl": ("Curl / flessione delle braccia", ["curl", "bicipiti"], ["tirata"]),
    "squat": ("Squat / accosciata", ["squat", "accosciata"], ["ginocchio", "anca"]),
    "leg_press": ("Pressa per le gambe", ["leg_press", "pressa"], ["ginocchio", "anca"]),
    "shoulder_press": ("Spinta sopra la testa", ["shoulder_press", "spalle"], ["spinta"]),
    "pulldown": ("Lat machine / tirata verticale", ["pulldown", "lat machine"], ["tirata"]),
    "row": ("Rematore / tirata orizzontale", ["row", "rematore"], ["tirata"]),
    "deadlift": ("Stacco / estensione delle anche", ["deadlift", "stacco"], ["anca"]),
    "sit_up": ("Sit-up / flessione del tronco", ["sit_up", "addominali"], ["core"]),
    "plank": ("Plank / tenuta del tronco", ["plank"], ["core"]),
    "push_up": ("Piegamenti sulle braccia", ["push_up", "piegamenti"], ["spinta", "core"]),
}


def normalize_categories(value) -> list[dict]:
    """Keep scalar/array order, unknowns and original values; enum is installed profile."""
    values = list(value) if isinstance(value, (list, tuple)) else [] if value is None else [value]
    enum = FIELD_TYPES["exercise_category"].enum
    result = []
    for index, original in enumerate(values):
        if isinstance(original, int) and not isinstance(original, bool):
            native = enum.get(original)
        elif isinstance(original, str) and original in enum.values():
            native = original
        else:
            native = None
        known = native is not None and native != "unknown"
        name, aliases, families = VOCABULARY.get(
            native,
            (
                f"Categoria non tradotta ({native})" if known else "Categoria non determinata",
                [],
                ["altro"],
            ),
        )
        result.append(
            {
                "index": index,
                "original_value": original,
                "category_code": original
                if isinstance(original, int) and not isinstance(original, bool)
                else next((code for code, name in enum.items() if name == native), None),
                "native_name": native,
                "known_category": known,
                "simple_name": name,
                "aliases": aliases,
                "movement_families": families if known else [],
                "provenance": "device_hint",
                "confirmed": False,
                "decoder": "fitdecode " + fitdecode.__version__ + " exercise_category enum",
                "vocabulary_version": VOCABULARY_VERSION,
            }
        )
    return result
