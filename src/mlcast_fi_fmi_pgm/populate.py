"""Fill the Zarr store with raster data, one shard per task.

Each shard is written in full by exactly one process, so no locking is needed
and partial-shard read-modify-write never happens. Finished shards are
recorded as marker files, which makes the run restartable.

Before reading and again before writing each shard, free space on the output
filesystem is checked. Below the threshold the shard is skipped (nothing
written, no marker) and the run ends with DiskSpaceError; rerun to resume.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import shutil
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


class DiskSpaceError(RuntimeError):
    """Free space on the output filesystem is below the safety threshold."""


def default_min_free(store: str | Path) -> int:
    """max(MIN_FREE_BYTES, MIN_FREE_FRACTION of the filesystem size)."""
    total = shutil.disk_usage(Path(store).parent).total
    return max(config.MIN_FREE_BYTES, int(config.MIN_FREE_FRACTION * total))


def check_free_space(store: str | Path, min_free: int) -> None:
    free = shutil.disk_usage(Path(store).parent).free
    if free < min_free:
        raise DiskSpaceError(
            f"only {free / 1e9:.0f} GB free next to {store}, need at least {min_free / 1e9:.0f} GB"
        )


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
    store, var_name, k, i0, i1, items, threads, min_free = args
    t_start = time.monotonic()
    try:
        check_free_space(store, min_free)
    except DiskSpaceError as err:
        return {"shard": k, "aborted": str(err)}
    ny, nx = config.HEIGHT, config.WIDTH
    block = np.full((i1 - i0, ny, nx), config.NODATA, dtype=np.uint8)
    failures = []
    with ThreadPoolExecutor(threads) as pool:
        for j, (data, err) in enumerate(pool.map(_read_one, items)):
            if err is None:
                block[j] = data
            else:
                failures.append({"time": str(items[j][0]), "path": items[j][1], "error": err})

    try:
        check_free_space(store, min_free)
    except DiskSpaceError as err:
        return {"shard": k, "aborted": str(err)}
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
    min_free: int | None = None,
) -> pd.DataFrame:
    """Write all (or selected) shards. Returns a table of unreadable files.

    Raises DiskSpaceError (after in-flight shards finish) if free space drops
    below ``min_free`` bytes (default: ``default_min_free(store)``).
    """
    store = Path(store)
    if min_free is None:
        min_free = default_min_free(store)
    check_free_space(store, min_free)
    log.info("Free-space guard: stop if below %.0f GB", min_free / 1e9)
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
        todo.append((str(store), var_name, k, i0, i1, items, threads, min_free))

    log.info("%d of %d shards to write (%d timesteps each)", len(todo), n_shards, shard_t)
    t0 = time.monotonic()
    aborted: list[dict] = []
    if todo:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for n, res in enumerate(pool.imap_unordered(_write_shard, todo), 1):
                if "aborted" in res:
                    if not aborted:
                        log.error("Stopping: %s", res["aborted"])
                    aborted.append(res)
                    continue
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
    if aborted:
        raise DiskSpaceError(
            f"{len(aborted)} shards skipped for lack of disk space: {aborted[0]['aborted']}. "
            "Free space and rerun the same command to resume."
        )
    return df
