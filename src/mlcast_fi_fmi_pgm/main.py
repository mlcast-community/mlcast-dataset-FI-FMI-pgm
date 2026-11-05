"""Command line entry point: inventory → verify → create → populate → radars."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from . import config
from .create import create_store, sources_path
from .inventory import build_time_axis, scan_archive, summarize
from .populate import DiskSpaceError, populate_store
from .radars import add_radars, summarize_radars
from .verify import bad_files_path, verify_files

log = logging.getLogger("mlcast_fi_fmi_pgm")

DEFAULT_INVENTORY = f"{config.OUTPUT_DIR}/inventory.parquet"
DEFAULT_STORE = f"{config.OUTPUT_DIR}/fi-fmi-{config.VARIABLE_NAME}-5min.zarr"


def cmd_inventory(args: argparse.Namespace) -> None:
    end = args.end or pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
    df = scan_archive(args.start, end, root=args.root, threads=args.threads)
    Path(args.inventory).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.inventory)
    log.info("Wrote %d entries to %s", len(df), args.inventory)
    print(summarize(build_time_axis(df)))


def _load_inventory(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_parquet(args.inventory)
    if args.start:
        df = df[df["time"] >= pd.Timestamp(args.start)]
    if args.end:
        df = df[df["time"] <= pd.Timestamp(args.end)]
    return df


def cmd_verify(args: argparse.Namespace) -> None:
    # Check every row, including duplicates, so a good copy can replace a bad one.
    bad = verify_files(_load_inventory(args), workers=args.workers, threads=args.threads)
    out = bad_files_path(args.inventory)
    bad.to_csv(out, index=False)
    log.info("%d bad files written to %s", len(bad), out)


def _load_axis(args: argparse.Namespace):
    bad_csv = bad_files_path(args.inventory)
    if bad_csv.exists():
        bad = set(pd.read_csv(bad_csv)["path"])
        log.info("Excluding %d files listed in %s", len(bad), bad_csv)
    elif args.allow_unverified:
        bad = set()
        log.warning("No %s; unreadable files will become all-NaN timesteps", bad_csv)
    else:
        raise SystemExit(f"{bad_csv} not found: run 'verify' first (or --allow-unverified)")
    return build_time_axis(_load_inventory(args), bad_paths=bad)


def cmd_create(args: argparse.Namespace) -> None:
    axis = _load_axis(args)
    print(summarize(axis))
    create_store(
        args.store,
        axis,
        var_name=args.var,
        shard_size=args.shard_size,
        clevel=args.clevel,
        license=args.license,
        created_by=args.created_by,
        dataset_version=args.dataset_version,
        overwrite=args.overwrite,
        creators=args.creators,
    )
    if not axis.dropped.empty:
        dropped = Path(str(args.store) + ".dropped.csv")
        axis.dropped.to_csv(dropped, index=False)
        log.info("Dropped inventory rows listed in %s", dropped)


def cmd_populate(args: argparse.Namespace) -> None:
    populate_store(
        args.store,
        var_name=args.var,
        workers=args.workers,
        threads=args.threads,
        redo=args.redo,
        shards=args.shards,
        min_free=None if args.min_free_gb is None else int(args.min_free_gb * 1e9),
    )


def cmd_radars(args: argparse.Namespace) -> None:
    ds = add_radars(args.store, workers=args.workers, threads=args.threads, rescan=args.rescan)
    times = pd.DatetimeIndex(pd.read_parquet(sources_path(args.store))["time"])
    print(summarize_radars(ds, times))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fmi-pgm-zarr", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, store=True):
        sp.add_argument("--inventory", default=DEFAULT_INVENTORY, help="inventory parquet")
        if store:
            sp.add_argument("--store", default=DEFAULT_STORE, help="output Zarr store")
            sp.add_argument("--var", default=config.VARIABLE_NAME, help="data variable name")

    sp = sub.add_parser("inventory", help="scan the archive into a parquet inventory")
    common(sp, store=False)
    sp.add_argument("--start", default="2008-01-01")
    sp.add_argument("--end", default=None, help="default: today 00:00 UTC")
    sp.add_argument("--root", default=config.ARCHIVE_ROOT)
    sp.add_argument("--threads", type=int, default=16)
    sp.set_defaults(func=cmd_inventory)

    sp = sub.add_parser("verify", help="read every file once and list unusable ones")
    common(sp, store=False)
    sp.add_argument("--start", default=None)
    sp.add_argument("--end", default=None)
    sp.add_argument("--workers", type=int, default=6, help="parallel processes")
    sp.add_argument("--threads", type=int, default=8, help="reader threads per process")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("create", help="create the empty Zarr store from the inventory")
    common(sp)
    sp.add_argument("--start", default=None)
    sp.add_argument("--end", default=None)
    sp.add_argument("--allow-unverified", action="store_true", help="create without verify")
    sp.add_argument("--shard-size", type=int, default=config.SHARD_SIZE)
    sp.add_argument("--clevel", type=int, default=config.CLEVEL)
    sp.add_argument("--license", default=config.LICENSE)
    sp.add_argument("--created-by", default=config.CREATED_BY, help="single 'Name <email>' contact")
    sp.add_argument(
        "--creators",
        nargs="+",
        default=list(config.CREATORS),
        metavar="'NAME <EMAIL>'",
        help="all dataset creators (ACDD creator_name/creator_email)",
    )
    sp.add_argument("--dataset-version", default=config.DATASET_VERSION)
    sp.add_argument("--overwrite", action="store_true")
    sp.set_defaults(func=cmd_create)

    sp = sub.add_parser("populate", help="write raster data into the store")
    common(sp)
    sp.add_argument("--workers", type=int, default=4, help="parallel shard processes")
    sp.add_argument("--threads", type=int, default=8, help="file reader threads per process")
    sp.add_argument("--redo", action="store_true", help="rewrite shards already marked done")
    sp.add_argument("--shards", type=int, nargs="*", default=None, help="only these shards")
    sp.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="stop before writing if free space would drop below this "
        "(default: max(500 GB, 10%% of the filesystem))",
    )
    sp.set_defaults(func=cmd_populate)

    sp = sub.add_parser(
        "radars", help="add radar_status (which radars contributed) from the file headers"
    )
    common(sp)
    sp.add_argument("--workers", type=int, default=6, help="parallel processes")
    sp.add_argument("--threads", type=int, default=8, help="header reader threads per process")
    sp.add_argument("--rescan", action="store_true", help="reread headers even if a scan exists")
    sp.set_defaults(func=cmd_radars)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        args.func(args)
    except DiskSpaceError as err:
        log.error("%s", err)
        raise SystemExit(2) from err


if __name__ == "__main__":
    main()
