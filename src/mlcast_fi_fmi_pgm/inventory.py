"""Scan the FMI archive and build a parquet inventory of SUOMI1 files."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


def _scan_day_dir(path: str) -> list[tuple[str, str, int]]:
    try:
        entries = list(os.scandir(path))
    except FileNotFoundError:
        return []
    out = []
    for entry in entries:
        m = config.FILENAME_RE.match(entry.name)
        if m:
            try:
                size = entry.stat().st_size
            except OSError:
                size = -1
            out.append((m.group(1), entry.path, size))
    return out


def scan_archive(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    root: str = config.ARCHIVE_ROOT,
    threads: int = 16,
) -> pd.DataFrame:
    """List SUOMI1 files with timestamps in [start, end].

    Directory dates do not always match file timestamps, so one extra day on
    each side is scanned and the timestamp is always taken from the filename.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    one_day = pd.Timedelta("1D")
    days = pd.date_range(start.normalize() - one_day, end.normalize() + one_day)
    dirs = [
        config.DAY_DIR_TEMPLATE.format(root=root, year=d.year, month=d.month, day=d.day)
        for d in days
    ]
    log.info("Scanning %d day directories under %s", len(dirs), root)
    rows: list[tuple[str, str, int]] = []
    with ThreadPoolExecutor(threads) as pool:
        for i, found in enumerate(pool.map(_scan_day_dir, dirs), 1):
            rows.extend(found)
            if i % 365 == 0:
                log.info("  %d/%d directories, %d files", i, len(dirs), len(rows))

    df = pd.DataFrame(rows, columns=["stamp", "path", "size"])
    df["time"] = pd.to_datetime(df["stamp"], format="%Y%m%d%H%M").astype("datetime64[ns]")
    df = df.drop(columns="stamp")
    df = df[(df["time"] >= start) & (df["time"] <= end)]
    return df.sort_values(["time", "path"]).reset_index(drop=True)[["time", "path", "size"]]


@dataclass
class TimeAxis:
    """Time coordinate and gaps derived from an inventory."""

    files: pd.DataFrame  # one row per kept timestep: time, path
    missing_times: np.ndarray  # datetime64[ns]
    dropped: pd.DataFrame  # rows excluded from the dataset, with a 'reason' column


def build_time_axis(
    inventory: pd.DataFrame,
    timestep: str = config.TIMESTEP,
    bad_paths: set[str] | frozenset[str] = frozenset(),
) -> TimeAxis:
    """Clean an inventory into a regular time axis with explicit missing times.

    Drops empty files, files listed in ``bad_paths`` (from ``verify``), times
    that are not on the regular timestep grid, and duplicate timestamps
    (keeping the first usable path).
    """
    inv = inventory.sort_values(["time", "path"]).reset_index(drop=True)
    step = pd.Timedelta(timestep)

    reasons = pd.Series("", index=inv.index)
    reasons[inv["size"] <= 0] = "empty or unreadable file"
    reasons[inv["path"].isin(bad_paths)] = "failed verification"
    reasons[(inv["time"] - inv["time"].dt.floor(step)) != pd.Timedelta(0)] = "off timestep grid"
    ok = reasons == ""
    dup = inv.loc[ok, "time"].duplicated(keep="first").reindex(inv.index, fill_value=False)
    reasons[dup] = "duplicate timestamp"
    ok &= ~dup

    files = inv[ok].reset_index(drop=True)
    dropped = inv[~ok].assign(reason=reasons[~ok]).reset_index(drop=True)

    if files.empty:
        missing = np.array([], dtype="datetime64[ns]")
    else:
        full = pd.date_range(files["time"].iloc[0], files["time"].iloc[-1], freq=step)
        missing = full.difference(pd.DatetimeIndex(files["time"])).values.astype("datetime64[ns]")
    return TimeAxis(files=files, missing_times=missing, dropped=dropped)


def summarize(axis: TimeAxis) -> str:
    files = axis.files
    if files.empty:
        return "No files."
    t = pd.DatetimeIndex(files["time"])
    expected = len(t) + len(axis.missing_times)
    lines = [
        f"Period: {t[0]} – {t[-1]}",
        f"Timesteps present: {len(t)} / {expected} ({100 * len(t) / expected:.2f} %)",
        f"Missing timesteps: {len(axis.missing_times)}",
        f"Dropped files: {len(axis.dropped)}",
    ]
    if not axis.dropped.empty:
        for reason, n in axis.dropped["reason"].value_counts().items():
            lines.append(f"  {reason}: {n}")
    per_year = pd.Series(1, index=t).groupby(t.year).sum()
    lines.append("Timesteps per year:")
    lines += [f"  {y}: {n}" for y, n in per_year.items()]
    return "\n".join(lines)
