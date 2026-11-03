"""Read every inventoried file once and list the ones that cannot be used.

Unreadable files must not end up in the `time` coordinate (their timesteps
belong in `missing_times`), so this runs before the store is created.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .pgm import PGMError, check_header, read_pgm

log = logging.getLogger(__name__)


def bad_files_path(inventory: str | Path) -> Path:
    inventory = Path(inventory)
    return inventory.with_name(inventory.stem + ".bad.csv")


def _check_one(item: tuple[pd.Timestamp, str]) -> str | None:
    t, path = item
    try:
        header, _ = read_pgm(path)
        check_header(header, expected_time=t)
    except PGMError as err:
        return str(err)
    except Exception as err:  # one odd file must not abort a multi-hour run
        return f"unexpected {type(err).__name__}: {err}"
    return None


def _check_batch(args: tuple[list[tuple[pd.Timestamp, str]], int]) -> list[dict]:
    items, threads = args
    with ThreadPoolExecutor(threads) as pool:
        errors = list(pool.map(_check_one, items))
    return [
        {"time": t, "path": p, "error": e} for (t, p), e in zip(items, errors, strict=True) if e
    ]


def verify_files(
    inventory: pd.DataFrame, workers: int = 6, threads: int = 8, batch: int = 2000
) -> pd.DataFrame:
    """Return a table (time, path, error) of files that fail to read or validate."""
    items = list(zip(inventory["time"], inventory["path"], strict=True))
    tasks = [(items[i : i + batch], threads) for i in range(0, len(items), batch)]
    log.info("Verifying %d files in %d batches", len(items), len(tasks))
    bad: list[dict] = []
    t0 = time.monotonic()
    with mp.get_context("spawn").Pool(workers) as pool:
        for n, found in enumerate(pool.imap_unordered(_check_batch, tasks), 1):
            bad += found
            if n % 50 == 0 or n == len(tasks):
                eta = (time.monotonic() - t0) / n * (len(tasks) - n)
                log.info("  %d/%d batches, %d bad, ETA %.0f min", n, len(tasks), len(bad), eta / 60)
    return pd.DataFrame(bad, columns=["time", "path", "error"]).sort_values("time")
