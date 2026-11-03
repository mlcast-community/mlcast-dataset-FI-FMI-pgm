"""Fill the Zarr store with raster data, one shard per task.

Each shard is written in full by exactly one process, so no locking is needed
and partial-shard read-modify-write never happens. Finished shards are
recorded as marker files, which makes the run restartable.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from . import config
from .create import sources_path
from .pgm import PGMError, check_header, read_pgm

log = logging.getLogger(__name__)


def progress_dir(store: str | Path) -> Path:
    store = Path(store)
    return store.with_name(store.name + ".progress")


def failures_path(store: str | Path) -> Path:
    store = Path(store)
    return store.with_name(store.name + ".failures.csv")


def _read_one(item: tuple[pd.Timestamp, str]) -> tuple[np.ndarray | None, str | None]:
    t, path = item
    try:
        header, data = read_pgm(path)
        check_header(header, expected_time=t)
        return data, None
    except PGMError as err:
        return None, str(err)
    except Exception as err:  # one odd file must not abort a multi-hour run
        return None, f"unexpected {type(err).__name__}: {err}"


def _write_shard(args: tuple) -> dict:
    store, var_name, k, i0, i1, items, threads = args
    t_start = time.monotonic()
    ny, nx = config.HEIGHT, config.WIDTH
    block = np.full((i1 - i0, ny, nx), config.NODATA, dtype=np.uint8)
    failures = []
    with ThreadPoolExecutor(threads) as pool:
        for j, (data, err) in enumerate(pool.map(_read_one, items)):
            if err is None:
                block[j] = data
            else:
                failures.append({"time": str(items[j][0]), "path": items[j][1], "error": err})

    arr = zarr.open_array(store, path=var_name, mode="r+")
    arr[i0:i1] = block

    result = {
        "shard": k,
        "start": i0,
        "stop": i1,
        "failures": failures,
        "seconds": round(time.monotonic() - t_start, 1),
    }
    marker = progress_dir(store) / f"shard_{k:06d}.done"
    marker.write_text(json.dumps(result))
    return result


def populate_store(
    store: str | Path,
    var_name: str = config.VARIABLE_NAME,
    workers: int = 4,
    threads: int = 8,
    redo: bool = False,
    shards: list[int] | None = None,
) -> pd.DataFrame:
    """Write all (or selected) shards. Returns a table of unreadable files."""
    store = Path(store)
    arr = zarr.open_array(store, path=var_name, mode="r")
    nt = arr.shape[0]
    shard_t = arr.shards[0] if arr.shards else arr.chunks[0]

    sources = pd.read_parquet(sources_path(store))
    if len(sources) != nt:
        raise RuntimeError(f"{sources_path(store)} has {len(sources)} rows, store has {nt}")
    times = list(sources["time"])
    paths = list(sources["path"])

    pdir = progress_dir(store)
    pdir.mkdir(exist_ok=True)
    n_shards = -(-nt // shard_t)
    todo = []
    for k in shards if shards is not None else range(n_shards):
        if not redo and (pdir / f"shard_{k:06d}.done").exists():
            continue
        i0, i1 = k * shard_t, min((k + 1) * shard_t, nt)
        items = list(zip(times[i0:i1], paths[i0:i1], strict=True))
        todo.append((str(store), var_name, k, i0, i1, items, threads))

    log.info("%d of %d shards to write (%d timesteps each)", len(todo), n_shards, shard_t)
    t0 = time.monotonic()
    if todo:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for n, res in enumerate(pool.imap_unordered(_write_shard, todo), 1):
                elapsed = time.monotonic() - t0
                eta = elapsed / n * (len(todo) - n)
                log.info(
                    "shard %d done (%ds, %d failures) — %d/%d, ETA %.1f h",
                    res["shard"],
                    res["seconds"],
                    len(res["failures"]),
                    n,
                    len(todo),
                    eta / 3600,
                )

    failures = []
    for marker in sorted(pdir.glob("shard_*.done")):
        failures += json.loads(marker.read_text())["failures"]
    df = pd.DataFrame(failures, columns=["time", "path", "error"])
    df.to_csv(failures_path(store), index=False)
    if len(df):
        log.warning("%d unreadable files written as fill; see %s", len(df), failures_path(store))
    return df
