"""Reader for FMI gzip-compressed binary PGM (P5) radar composites."""

from __future__ import annotations

import gzip
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


class PGMError(ValueError):
    """Raised when a file is not a valid FMI PGM composite."""


@dataclass
class PGMHeader:
    width: int
    height: int
    maxval: int
    meta: dict[str, str] = field(default_factory=dict)
    radars: list[str] = field(default_factory=list)

    @property
    def obstime(self) -> pd.Timestamp | None:
        value = self.meta.get("obstime")
        return pd.Timestamp(value) if value else None

    def corner(self, key: str) -> tuple[float, float] | None:
        value = self.meta.get(key)
        if value is None:
            return None
        lon, lat = (float(v) for v in value.split())
        return lon, lat


def _parse(raw: bytes) -> tuple[PGMHeader, np.ndarray]:
    if raw[:2] != b"P5":
        raise PGMError(f"not a binary PGM (magic {raw[:2]!r})")

    pos = 2
    numbers: list[int] = []
    comments: list[str] = []
    n = len(raw)
    while len(numbers) < 3:
        while pos < n and raw[pos : pos + 1].isspace():
            pos += 1
        if pos >= n:
            raise PGMError("truncated header")
        if raw[pos : pos + 1] == b"#":
            end = raw.find(b"\n", pos)
            if end == -1:
                raise PGMError("truncated header comment")
            comments.append(raw[pos + 1 : end].decode("ascii", "replace").strip())
            pos = end + 1
            continue
        start = pos
        while pos < n and not raw[pos : pos + 1].isspace():
            pos += 1
        try:
            numbers.append(int(raw[start:pos]))
        except ValueError as err:
            raise PGMError(f"bad header token {raw[start:pos]!r}") from err
    pos += 1  # exactly one whitespace character separates header and raster

    width, height, maxval = numbers
    if maxval > 255:
        raise PGMError(f"maxval {maxval} > 255 not supported")

    meta: dict[str, str] = {}
    radars: list[str] = []
    for line in comments:
        key, _, value = line.partition(" ")
        if key == "radar":
            radars.append(value.strip())
        elif key and key != "}":
            meta[key] = value.strip()

    data = np.frombuffer(raw, dtype=np.uint8, count=width * height, offset=pos)
    if len(raw) - pos != width * height:
        raise PGMError(f"raster size {len(raw) - pos} != {width}x{height}")
    header = PGMHeader(width, height, maxval, meta, radars)
    return header, data.reshape(height, width)


def read_pgm(path: str | Path) -> tuple[PGMHeader, np.ndarray]:
    """Read a (gzipped) PGM file and return its header and uint8 raster.

    Row 0 of the raster is the northern edge of the grid.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rb") as fh:
            raw = fh.read()
    except (OSError, EOFError, zlib.error) as err:
        raise PGMError(f"cannot read {path}: {err}") from err
    return _parse(raw)


def check_header(header: PGMHeader, expected_time: pd.Timestamp | None = None) -> None:
    """Raise PGMError if the header does not describe the SUOMI1 grid."""
    if (header.width, header.height) != (config.WIDTH, config.HEIGHT):
        raise PGMError(f"unexpected size {header.width}x{header.height}")
    if header.maxval != config.MAXVAL:
        raise PGMError(f"unexpected maxval {header.maxval}")
    for key, expected in (
        ("bottomleft", config.BOTTOM_LEFT_LONLAT),
        ("topright", config.TOP_RIGHT_LONLAT),
    ):
        corner = header.corner(key)
        if corner is not None and not np.allclose(corner, expected, atol=1e-3):
            raise PGMError(f"unexpected {key} {corner}")
    if expected_time is not None and header.obstime is not None:
        if header.obstime != expected_time:
            raise PGMError(f"obstime {header.obstime} != filename time {expected_time}")
