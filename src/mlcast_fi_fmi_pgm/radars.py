"""Which radars contributed to each timestep, from the PGM headers.

Every header lists the radars of the composite, one line each:
``# radar <code> <flag> <lon> <lat>``. The flag is 1 when the radar
contributed and 0 when it is part of the network but did not contribute
(checked against the data: with flag 0 the radar's own coverage turns into
nodata and nothing else changes).

The scan reads only the first few KB of every source file and saves a long
table to ``<store>.radars.parquet``; ``write_radar_variables`` turns it into
``radar_status(time, radar)`` plus ``radar_lon``/``radar_lat`` coordinates and
appends them to the existing store without touching the data variable.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from . import config
from .create import sources_path
from .pgm import PGMError, read_pgm_header

log = logging.getLogger(__name__)

NOT_LISTED, NOT_CONTRIBUTING, CONTRIBUTING = -1, 0, 1
STATUS_VAR = "radar_status"
RADAR_VARS = (STATUS_VAR, "radar", "radar_lon", "radar_lat")
COLUMNS = ["index", "radar", "flag", "lon", "lat"]


def radars_path(store: str | Path) -> Path:
    store = Path(store)
    return store.with_name(store.name + ".radars.parquet")


def radar_errors_path(store: str | Path) -> Path:
    store = Path(store)
    return store.with_name(store.name + ".radars.errors.csv")


def parse_radar_lines(lines: list[str]) -> list[tuple[str, int, float, float]]:
    """``"KOR 1 21.6460 60.1280"`` → ``("KOR", 1, 21.646, 60.128)``."""
    out = []
    for line in lines:
        fields = line.split()
        if len(fields) < 4:
            raise PGMError(f"bad radar line {line!r}")
        code, flag, lon, lat = fields[:4]
        if flag not in ("0", "1"):
            raise PGMError(f"unexpected radar flag in {line!r}")
        out.append((code, int(flag), float(lon), float(lat)))
    return out


def _scan_one(item: tuple[int, str]) -> tuple[list[tuple], str | None]:
    i, path = item
    try:
        radars = parse_radar_lines(read_pgm_header(path).radars)
    except PGMError as err:
        return [], str(err)
    except Exception as err:  # one odd file must not abort a long run
        return [], f"unexpected {type(err).__name__}: {err}"
    return [(i, *r) for r in radars], None


def _scan_batch(args: tuple[list[tuple[int, str]], int]) -> tuple[list[tuple], list[dict]]:
    items, threads = args
    rows, errors = [], []
    with ThreadPoolExecutor(threads) as pool:
        for (i, path), (found, err) in zip(items, pool.map(_scan_one, items), strict=True):
            rows += found
            if err:
                errors.append({"index": i, "path": path, "error": err})
    return rows, errors


def scan_radars(
    paths: list[str], workers: int = 6, threads: int = 8, batch: int = 5000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the radar list of every file.

    Returns (rows, errors): one row per (time index, radar) with columns
    ``index, radar, flag, lon, lat``, and a table of files whose header could
    not be read.
    """
    items = list(enumerate(paths))
    tasks = [(items[i : i + batch], threads) for i in range(0, len(items), batch)]
    log.info("Reading radar lists from %d headers in %d batches", len(items), len(tasks))
    rows: list[tuple] = []
    errors: list[dict] = []
    t0 = time.monotonic()
    with mp.get_context("spawn").Pool(workers) as pool:
        for n, (found, errs) in enumerate(pool.imap_unordered(_scan_batch, tasks), 1):
            rows += found
            errors += errs
            if n % 20 == 0 or n == len(tasks):
                eta = (time.monotonic() - t0) / n * (len(tasks) - n)
                log.info("  %d/%d batches, ETA %.0f min", n, len(tasks), eta / 60)
    df = pd.DataFrame(rows, columns=COLUMNS).astype(
        {"index": "int32", "radar": "category", "flag": "int8", "lon": "float32", "lat": "float32"}
    )
    df = df.sort_values(["index", "radar"], ignore_index=True)
    return df, pd.DataFrame(errors, columns=["index", "path", "error"])


def _positions(rows: pd.DataFrame, times: pd.DatetimeIndex) -> tuple[pd.DataFrame, list[str]]:
    """Most frequent position of each radar, plus notes on any other positions."""
    counts = rows.groupby(["radar", "lon", "lat"], observed=True).size().rename("n").reset_index()
    counts = counts.sort_values(["radar", "n"], ascending=[True, False])
    main = counts.drop_duplicates("radar").set_index("radar")[["lon", "lat"]]
    notes = []
    for row in counts[counts.duplicated("radar")].itertuples():
        sel = rows.loc[
            (rows["radar"] == row.radar) & (rows["lon"] == row.lon) & (rows["lat"] == row.lat),
            "index",
        ]
        notes.append(
            f"{row.radar} is listed at {row.lon:.4f}E {row.lat:.4f}N in {row.n} files "
            f"between {times[sel.min()]:%Y-%m-%d %H:%M} and {times[sel.max()]:%Y-%m-%d %H:%M}"
        )
    return main, notes


def build_radar_dataset(rows: pd.DataFrame, times: pd.DatetimeIndex, attrs: dict) -> xr.Dataset:
    """``radar_status(time, radar)`` with ``radar_lon``/``radar_lat`` coordinates."""
    dup = rows.duplicated(["index", "radar"])
    if dup.any():
        log.warning("%d repeated radar lines within a header; keeping the first", dup.sum())
        rows = rows[~dup]
    codes = sorted(rows["radar"].astype(str).unique())
    status = np.full((len(times), len(codes)), NOT_LISTED, dtype=np.int8)
    col = pd.Categorical(rows["radar"].astype(str), categories=codes).codes
    status[rows["index"].to_numpy(), col] = rows["flag"].to_numpy()

    pos, notes = _positions(rows, times)
    pos = pos.reindex(codes)
    comment = (
        "From the '# radar <code> <flag> <lon> <lat>' lines of each source file header. "
        "Flag 1: the radar contributed to the composite. Flag 0: the radar is part of the "
        "network but did not contribute (verified on the data: its own coverage area is "
        "nodata and nothing else changes). Radars that are not listed are -1."
    )
    pos_comment = "Most frequent position in the source headers."
    if notes:
        pos_comment += " Other listed positions: " + "; ".join(notes) + "."
    # radar_status is a coordinate, not a data variable: the MLCast spec
    # applies its (time, y, x) / float / grid_mapping rules to every data var.
    ds = xr.Dataset(
        coords={
            STATUS_VAR: (
                ("time", "radar"),
                status,
                {
                    "long_name": "Contribution of each weather radar to the composite",
                    "flag_values": np.array(
                        [NOT_LISTED, NOT_CONTRIBUTING, CONTRIBUTING], dtype=np.int8
                    ),
                    "flag_meanings": "not_listed not_contributing contributing",
                    "comment": comment,
                },
            ),
            "radar": ("radar", np.array(codes), {"long_name": "FMI radar site code"}),
            "radar_lon": (
                "radar",
                pos["lon"].to_numpy(np.float32),
                {
                    "long_name": "radar longitude",
                    "units": "degrees_east",
                    "comment": pos_comment,
                },
            ),
            "radar_lat": (
                "radar",
                pos["lat"].to_numpy(np.float32),
                {
                    "long_name": "radar latitude",
                    "units": "degrees_north",
                    "comment": pos_comment,
                },
            ),
        },
        attrs=attrs,
    )
    return ds


def write_radar_variables(
    store: str | Path, rows: pd.DataFrame, chunk: int = config.RADAR_CHUNK
) -> xr.Dataset:
    """Append (or replace) the radar variables in an existing store."""
    store = Path(store)
    existing = xr.open_zarr(store)
    times = pd.DatetimeIndex(existing["time"].values)
    if len(rows) and rows["index"].max() >= len(times):
        raise ValueError("radar table does not match the store's time axis")
    # mode="a" replaces the group attributes with the dataset's, so carry them over.
    ds = build_radar_dataset(rows, times, dict(existing.attrs))
    existing.close()

    group = zarr.open_group(store, mode="r+")
    for name in RADAR_VARS:
        if name in group:
            del group[name]
    nr = ds.sizes["radar"]
    ds.to_zarr(
        store,
        mode="a",
        zarr_format=3,
        consolidated=True,
        encoding={
            # No _FillValue: -1 is a real flag value, not missing data, and
            # the variable must stay integer when decoded.
            STATUS_VAR: {"fill_value": NOT_LISTED, "chunks": (min(chunk, len(times)), nr)},
        },
    )
    return ds


def summarize_radars(ds: xr.Dataset, times: pd.DatetimeIndex) -> str:
    """One line per radar: first/last time listed, timesteps with flag 1 and 0."""
    status = ds[STATUS_VAR]
    lines = [f"{'radar':5} {'first listed':16} {'last listed':16} {'contrib.':>8} {'flag 0':>8}"]
    for code in ds["radar"].values:
        s = status.sel(radar=code).values
        listed = np.flatnonzero(s != NOT_LISTED)
        lines.append(
            f"{code:5} {times[listed[0]]:%Y-%m-%d %H:%M} {times[listed[-1]]:%Y-%m-%d %H:%M} "
            f"{(s == CONTRIBUTING).sum():8d} {(s == NOT_CONTRIBUTING).sum():8d}"
        )
    return "\n".join(lines)


def add_radars(
    store: str | Path,
    workers: int = 6,
    threads: int = 8,
    rescan: bool = False,
    chunk: int = config.RADAR_CHUNK,
) -> xr.Dataset:
    """Scan the headers (unless a scan exists) and write the radar variables."""
    store = Path(store)
    out = radars_path(store)
    if out.exists() and not rescan:
        log.info("Using existing scan %s (--rescan to redo)", out)
        rows = pd.read_parquet(out)
    else:
        sources = pd.read_parquet(sources_path(store))
        rows, errors = scan_radars(list(sources["path"]), workers=workers, threads=threads)
        errors.to_csv(radar_errors_path(store), index=False)
        if len(errors):
            raise RuntimeError(
                f"{len(errors)} headers could not be read; see {radar_errors_path(store)}. "
                "Nothing was written."
            )
        rows.to_parquet(out)
        log.info("Wrote %d radar rows to %s", len(rows), out)
    ds = write_radar_variables(store, rows, chunk=chunk)
    log.info("Wrote %s for %d radars to %s", STATUS_VAR, ds.sizes["radar"], store)
    return ds
