"""Bounded FIT / single-member ZIP ingestion, without filesystem extraction."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile
from zlib import error as ZlibError

from .models import DataError
from .strength_fit import read_strength_fit

MAX_INPUT_BYTES = 10 * 1024 * 1024


def unpack_strength_input(content: bytes) -> tuple[bytes, dict]:
    if not 0 < len(content) <= MAX_INPUT_BYTES:
        raise DataError("FIT/ZIP vuoto o oltre 10 MiB")
    descriptor = {"input_sha256": hashlib.sha256(content).hexdigest(), "kind": "fit"}
    if not content.startswith(b"PK"):
        return content, descriptor
    try:
        with ZipFile(io.BytesIO(content)) as archive:
            # Directory entries are unnecessary for Garmin single-FIT exports.
            members = archive.infolist()
            if len(members) != 1 or members[0].is_dir():
                raise DataError("ZIP richiesto con un solo file FIT")
            member = members[0]
            name = member.filename.replace("\\", "/")
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or ":" in name
                or path.suffix.lower() != ".fit"
                or member.flag_bits & 1
                or member.external_attr >> 16 & 0o170000 == 0o120000
            ):
                raise DataError("Membro ZIP non consentito: richiesto FIT normale non cifrato")
            if not 0 < member.file_size <= MAX_INPUT_BYTES:
                raise DataError("FIT decompresso vuoto o oltre 10 MiB")
            with archive.open(member) as stream:
                decoded = stream.read(MAX_INPUT_BYTES + 1)
            if len(decoded) != member.file_size or len(decoded) > MAX_INPUT_BYTES:
                raise DataError("Dimensione FIT decompresso non valida")
            descriptor.update(kind="zip", zip_crc_verified=True)
            return decoded, descriptor
    except (BadZipFile, ZlibError, RuntimeError, EOFError, OSError, NotImplementedError) as exc:
        raise DataError("ZIP non leggibile o CRC non valido") from exc


def read_strength_source(path: Path, *, synthetic: bool = False) -> tuple[dict, dict]:
    if not 0 < path.stat().st_size <= MAX_INPUT_BYTES:
        raise DataError("FIT/ZIP vuoto o oltre 10 MiB")
    content, descriptor = unpack_strength_input(path.read_bytes())
    with TemporaryDirectory(prefix="hta-strength-") as directory:
        source = Path(directory) / "input.fit"
        source.write_bytes(content)
        raw = read_strength_fit(source, synthetic=synthetic)
    return raw, descriptor
