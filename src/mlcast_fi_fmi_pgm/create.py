"""Create the empty Zarr v3 store (coordinates, metadata, unwritten data array)."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from zarr.codecs import ZstdCodec

from . import config
from .grid import Grid
from .inventory import TimeAxis

log = logging.getLogger(__name__)

TIME_ENCODING = {
    "units": "minutes since 1970-01-01 00:00:00",
    "calendar": "proleptic_gregorian",
    "dtype": "int64",
}


def sources_path(store: str | Path) -> Path:
    """Parquet listing the source file of every time index in the store."""
    store = Path(store)
    return store.with_name(store.name + ".sources.parquet")


def build_dataset(
    axis: TimeAxis,
    grid: Grid,
    var_name: str = config.VARIABLE_NAME,
    shard_size: int = config.SHARD_SIZE,
    attrs: dict | None = None,
) -> xr.Dataset:
    times = axis.files["time"].to_numpy().astype("datetime64[ns]")
    nt, ny, nx = len(times), grid.height, grid.width
    lat, lon = grid.latlon()

    data = da.zeros((nt, ny, nx), dtype=np.float32, chunks=(min(shard_size, nt), ny, nx))
    ds = xr.Dataset(
        {
            var_name: (
                ("time", "y", "x"),
                data,
                {
                    "long_name": "Equivalent reflectivity factor, lowest CAPPI composite",
                    "standard_name": "equivalent_reflectivity_factor",
                    "units": "dBZ",
                    "grid_mapping": "spatial_ref",
                    "comment": (
                        f"Raw value 0 ({config.OFFSET:g} dBZ) means no detected echo; "
                        "NaN means outside radar coverage."
                    ),
                    "source_param": "CorrectedReflectivity",
                },
            ),
            "missing_times": (
                ("missing_times",),
                axis.missing_times,
                {"long_name": "Timesteps with no source data in the regular 5-minute series"},
            ),
        },
        coords={
            "time": ("time", times, {"standard_name": "time", "axis": "T"}),
            "y": (
                "y",
                grid.y,
                {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y"},
            ),
            "x": (
                "x",
                grid.x,
                {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X"},
            ),
            "lat": (
                ("y", "x"),
                lat,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "lon": (
                ("y", "x"),
                lon,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "spatial_ref": ((), np.int64(0), grid.spatial_ref_attrs(lat, lon)),
        },
        attrs=attrs or {},
    )
    return ds


_CONTACT_RE = re.compile(r"^\s*([^<>,]+?)\s*<([^<>\s,]+@[^<>\s,]+)>\s*$")


def creator_attrs(creators: Sequence[str]) -> dict:
    """ACDD creator attributes from "Name <email>" entries."""
    names, emails = [], []
    for entry in creators:
        m = _CONTACT_RE.match(entry)
        if not m:
            raise ValueError(f"creator {entry!r} is not in 'Name <email>' format")
        names.append(m.group(1))
        emails.append(m.group(2))
    return {
        "creator_name": ", ".join(names),
        "creator_email": ", ".join(emails),
        "creator_institution": config.INSTITUTION,
    }


def global_attrs(
    times: pd.DatetimeIndex,
    var_name: str,
    license: str,
    created_by: str,
    dataset_version: str,
    creators: Sequence[str] = config.CREATORS,
) -> dict:
    return {
        "title": "FMI radar reflectivity composite (lowest CAPPI, SUOMI1 grid)",
        "institution": config.INSTITUTION,
        "source": "FMI operational weather radar network composite",
        "Conventions": "CF-1.10",
        "license": license,
        **creator_attrs(creators),
        "time_coverage_start": times[0].isoformat(),
        # The validator treats gaps as a variable timestep and then requires this;
        # the series is regular (with missing_times) from the very first step.
        "consistent_timestep_start": times[0].isoformat(),
        "time_coverage_end": times[-1].isoformat(),
        "mlcast_created_on": pd.Timestamp.now(tz="UTC").isoformat(),
        "mlcast_created_by": created_by,
        "mlcast_created_with": f"{config.REPO_URL}@v{dataset_version}",
        "mlcast_dataset_version": dataset_version,
        "mlcast_dataset_validator_version": config.VALIDATOR_VERSION,
        "mlcast_dataset_identifier": f"FI-FMI-{var_name}",
    }


def create_store(
    store: str | Path,
    axis: TimeAxis,
    var_name: str = config.VARIABLE_NAME,
    shard_size: int = config.SHARD_SIZE,
    clevel: int = config.CLEVEL,
    license: str = config.LICENSE,
    created_by: str = config.CREATED_BY,
    dataset_version: str = config.DATASET_VERSION,
    overwrite: bool = False,
    creators: Sequence[str] = config.CREATORS,
) -> xr.Dataset:
    """Write coordinates and metadata; the data array is left unwritten (all fill)."""
    store = Path(store)
    if store.exists() and not overwrite:
        raise FileExistsError(f"{store} exists (use --overwrite to replace it)")
    if axis.files.empty:
        raise ValueError("no timesteps to write")

    grid = Grid()
    times = pd.DatetimeIndex(axis.files["time"])
    attrs = global_attrs(times, var_name, license, created_by, dataset_version, creators)
    ds = build_dataset(axis, grid, var_name, shard_size, attrs)

    nt = ds.sizes["time"]
    shard_t = min(shard_size, nt)
    encoding = {
        var_name: {
            "dtype": "uint8",
            "scale_factor": config.GAIN,
            "add_offset": config.OFFSET,
            "_FillValue": config.NODATA,
            # Zarr v3: the array's own fill_value is separate from the CF
            # _FillValue attribute and defaults to 0 (= -32 dBZ, "no echo").
            # It must be NODATA so unwritten chunks read as NaN, not dry weather.
            "fill_value": config.NODATA,
            "chunks": (1, grid.height, grid.width),
            "shards": (shard_t, grid.height, grid.width),
            # Plain zstd: on uint8 radar data it beats Blosc+bitshuffle in both
            # ratio (7.4 vs 6.5) and speed (8 vs 350 ms/step at level 9).
            "compressors": (ZstdCodec(level=clevel),),
        },
        "time": {**TIME_ENCODING, "chunks": (nt,)},
        "missing_times": {**TIME_ENCODING, "chunks": (max(len(axis.missing_times), 1),)},
        "lat": {"chunks": (grid.height, grid.width)},
        "lon": {"chunks": (grid.height, grid.width)},
    }

    log.info("Creating %s: %d timesteps, shards of %d", store, nt, shard_t)
    ds.to_zarr(
        store,
        mode="w",
        zarr_format=3,
        consolidated=True,
        compute=False,
        encoding=encoding,
    )
    axis.files[["time", "path"]].to_parquet(sources_path(store))
    return ds
