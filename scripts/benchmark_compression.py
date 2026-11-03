"""Benchmark chunk compression codecs on a seasonal sample of SUOMI1 timesteps.

Each timestep is one uint8 1226x760 chunk, as stored in the Zarr array.
Timings are single-threaded (Blosc threads = 1) so codecs compare fairly.

    uv run python scripts/benchmark_compression.py [--per-month 6] [--years ...]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from numcodecs import Blosc, Zstd

from mlcast_fi_fmi_pgm import config
from mlcast_fi_fmi_pgm.inventory import build_time_axis
from mlcast_fi_fmi_pgm.pgm import read_pgm
from mlcast_fi_fmi_pgm.verify import bad_files_path

N_TIMESTEPS_FULL = None  # filled from the time axis

SHUFFLES = {"noshuffle": Blosc.NOSHUFFLE, "shuffle": Blosc.SHUFFLE, "bitshuffle": Blosc.BITSHUFFLE}


def codecs() -> dict[str, object]:
    out: dict[str, object] = {f"zstd {lvl}": Zstd(level=lvl) for lvl in (3, 5, 9, 15, 19)}
    for clevel in (3, 5, 7, 9):
        for sname, s in SHUFFLES.items():
            out[f"blosc-zstd {clevel} {sname}"] = Blosc("zstd", clevel, s)
    return out


def season(month: int) -> str:
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA"}.get(month, "SON")  # fmt: skip


def _bench(args: tuple[str, np.ndarray]) -> dict:
    name, stack = args
    Blosc.use_threads = False
    codec = codecs()[name]
    sizes = np.empty(len(stack), dtype=np.int64)
    encoded = []
    t0 = time.perf_counter()
    for i, x in enumerate(stack):
        b = codec.encode(x)
        sizes[i] = len(b)
        encoded.append(b)
    t_enc = time.perf_counter() - t0
    t0 = time.perf_counter()
    for b in encoded:
        codec.decode(b)
    t_dec = time.perf_counter() - t0
    return {"codec": name, "sizes": sizes, "enc_ms": 1000 * t_enc / len(stack),
            "dec_ms": 1000 * t_dec / len(stack)}  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", default=f"{config.OUTPUT_DIR}/inventory.parquet")
    p.add_argument("--years", type=int, nargs="+", default=[2010, 2014, 2018, 2022, 2025])
    p.add_argument("--per-month", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    inv = pd.read_parquet(args.inventory)
    bad = set(pd.read_csv(bad_files_path(args.inventory))["path"])
    files = build_time_axis(inv, bad_paths=bad).files
    n_full = len(files)
    t = files["time"].dt
    sample = (
        files[t.year.isin(args.years)]
        .groupby([t.year, t.month], group_keys=False)
        .apply(lambda g: g.sample(min(args.per_month, len(g)), random_state=args.seed))
        .sort_values("time")
    )
    print(f"Sample: {len(sample)} timesteps, years {args.years}, {args.per_month}/month")

    with ThreadPoolExecutor(16) as pool:
        stack = np.stack([d for _, d in pool.map(read_pgm, sample["path"])])
    raw = stack[0].nbytes
    seasons = np.array([season(m) for m in sample["time"].dt.month])
    nodata = (stack == config.NODATA).mean()
    echo = ((stack != config.NODATA) & (stack != 0)).mean()
    print(f"Pixels: {100 * nodata:.1f} % nodata, {100 * echo:.1f} % with echo (value > 0)\n")

    names = list(codecs())
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(_bench, [(n, stack) for n in names])

    rows = []
    for r in results:
        s = r["sizes"]
        row = {
            "codec": r["codec"],
            "ratio": raw * len(s) / s.sum(),
            "enc ms": r["enc_ms"],
            "dec ms": r["dec_ms"],
            "archive GB": s.mean() * n_full / 1e9,
        }
        for sea in ("DJF", "MAM", "JJA", "SON"):
            row[f"ratio {sea}"] = raw * (seasons == sea).sum() / s[seasons == sea].sum()
        rows.append(row)
    df = pd.DataFrame(rows).set_index("codec")
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(df.to_string())
    print(f"\n'archive GB' = mean compressed chunk size x {n_full} timesteps (data only).")


if __name__ == "__main__":
    main()
