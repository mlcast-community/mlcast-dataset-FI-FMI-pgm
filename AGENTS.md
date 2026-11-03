# AGENTS.md

Guidance for AI coding agents working in this repository.

## Goal

Build a **Zarr v3** archive of the FMI (Finnish Meteorological Institute) radar
composite for the [MLCast](https://github.com/mlcast-community) dataset
collection. Tracking issue:
<https://github.com/mlcast-community/mlcast-datasets/issues/64> ("Adding the
fi-FMI dataset").

Reference implementation to follow for structure and conventions:
<https://github.com/mlcast-community/mlcast-dataset-BE-RMI-radclim/>
(inventory parquet → empty Zarr → populate, Python in `src/`, managed with
`uv`). Use **Zarr v3** (not v2): v2 produces too many files for HPC
filesystems. Use sharding to keep the file count low.

## Source data

- Root (FMI internal network only):
  `/arch/radar/storage/{year}/{month:02d}/{day:02d}/fmi/radar/composite/lowest`
- Filename:
  `{year}{month:02d}{day:02d}{hour:02d}{minute:02d}_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz`
- Only the `FIN_SUOMI1.pgm.gz` product is in scope. The same directory also
  holds `SUOMI1BIG`, `RAVAKE`, `RAVAKEBIG` and `.png` files; ignore them
  (match the filename exactly, not with a loose glob like `*SUOMI1*`).
- **Some files are not gzipped**: plain `…_FIN_SUOMI1.pgm` (all of
  2023-01..09, everything from 2026-09-19, and scattered files in other
  years). The filename regex accepts both; the reader picks by suffix.
- Timestep 5 min (288 files/day when complete). Coverage (inventory of
  2026-11-03): 2008-03-18 05:00 – 2026-11-02, 1 940 956 of 1 959 205
  timesteps (99.07 %).
- 162 timestamps exist in two neighbouring day directories; the copies are
  byte-identical, so the first usable one is kept.
- **Some archived files are corrupt** (e.g. all zero bytes, broken gzip
  streams), mostly in 2008. The `verify` step reads every file once and
  lists them so their timesteps go to `missing_times` instead of becoming
  all-NaN fields in `time`.
- **The directory date does not always match the file timestamp.** For
  example `2015/06/01/` contains `201505312340` … `201506012335`. Always take
  the timestamp from the filename (or the `# obstime` header), never from the
  directory, and search neighbouring day directories when building the
  inventory. Timestamps are UTC.

### File format

gzip-compressed binary PGM (`P5`), 760 (width) × 1226 (height), maxval 255,
uint8. ASCII `#` comment lines between the magic and the dimensions carry
metadata, e.g.:

```
# composite_area FIN
# projection_name SUOMI1
# radar KOR 1 21.6460 60.1280        (one line per contributing radar)
# obstime 202406010000
# producttype CAPPI
# productname LOWEST
# param CorrectedReflectivity
# metersperpixel_x 999.674053
# metersperpixel_y 999.62859
# projection radar {
# type stereographic
# centrallongitude 25
# centrallatitude 90
# truelatitude 60
# bottomleft 18.600000 57.930000
# topright 34.903000 69.005000
# }
```

Parse the header properly (comment lines can vary over the years, e.g. the
radar list changes). Don't hard-code a byte offset; read the last
`760*1226` bytes only after checking the dimensions and maxval.

Pixel values (observed on a 2024 sample):
- `255` – nodata / outside radar coverage
- `0` – no echo / lowest class
- `1–254` – reflectivity. FMI convention is `dBZ = 0.5 * value - 32`.

### Grid and projection

- Polar stereographic, `lon_0=25`, `lat_0=90`, `lat_ts=60`, spherical earth.
- proj string
  `+proj=stere +a=6371288 +lon_0=25E +lat_0=90N +lat_ts=60 +x_0=380886.310 +y_0=3395677.920 +no_defs`.
- Pixel size ≈ 1 km (999.67 m × 999.63 m).
- Corner lon/lat: bottom-left (18.600, 57.930), top-right (34.903, 69.005).
  These are the **outer pixel edges** (verified: they project to (0, 0) and
  (760·999.674, 1226·999.629) m exactly). Coordinates are pixel centres.
- PGM row 0 is the **northern** edge (verified against radar coverage), so
  `y` decreases with row index.
- Files from 2009–2014 state `metersperpixel_x/y` as 1015.38/1026.63, but
  corners and dimensions are identical, so it is the same grid; the old
  values are wrong metadata. Validate corners and size, not
  `metersperpixel_*`.

## Repository layout

Package `src/mlcast_fi_fmi_pgm/` (hatchling build, CLI `fmi-pgm-zarr`):

- `config.py` – archive paths, grid constants, scaling, output defaults.
- `pgm.py` – PGM reader and header validation (`check_header`).
- `grid.py` – projected x/y, 2D lat/lon, CF grid mapping + WKT (with BBOX).
- `inventory.py` – archive scan → parquet (`time, path, size`); builds the
  time axis (drops empty / duplicate / off-grid files, lists missing times).
- `create.py` – writes the empty Zarr v3 store (coords + metadata only) and
  `<store>.sources.parquet` (source path for each time index).
- `populate.py` – writes data one shard per task (process pool, reader
  threads inside); restartable via `<store>.progress/shard_*.done` markers;
  unreadable files are written as fill and listed in `<store>.failures.csv`.
- `verify.py` – reads every inventoried file once; writes
  `inventory.bad.csv` (time, path, error). `create` refuses to run without
  it unless `--allow-unverified`.
- `main.py` – CLI: `inventory`, `verify`, `create`, `populate` subcommands.
- `tests/` – pytest, with one sample `.pgm.gz` in `tests/data/`.

Pipeline (defaults write to `/tutka/data/storage/mlcast-dataset-fi-fmi-pgm/`):

```bash
uv run fmi-pgm-zarr inventory --start 2008-01-01 --end 2026-11-02
uv run fmi-pgm-zarr verify --workers 6 --threads 8
uv run fmi-pgm-zarr create            # --start/--end to subset, --overwrite
uv run fmi-pgm-zarr populate --workers 6 --threads 8
uv run mlcast.validate_dataset source_data radar_precipitation <store>
```

Validation needs two environments to cover all checks:

- the uv env (above) runs everything except GDAL (system GDAL 3.0.4 is too
  old to build Python 3.12 bindings), including the cartopy checks;
- `/tutka/code/dev/miniforge3/envs/jupyter` (shared conda env; GDAL 3.13 +
  `osgeo` bindings, rioxarray, zarr, validator 0.4.0 installed; no cartopy)
  runs the GDAL checks:
  `/tutka/code/dev/miniforge3/envs/jupyter/bin/mlcast.validate_dataset source_data radar_precipitation <store>`

Don't install further packages into that shared env without asking.

Keep dataset choices (chunking, compression, time range) configurable rather
than hard-coded.

## MLCast spec (validator `mlcast-dataset-validator` 0.4.0)

Key requirements that shape the design (read the spec in the validator's
`specs/source_data/radar_precipitation.py` before changing them):

- `time` contains **only timesteps with data**; gaps go in a separate
  `missing_times` variable and must not appear in `time`.
- Chunks are `1 × height × width`; we shard along time (default 2016
  timesteps = 1 week per shard file). Plain zstd level 9 (`ZstdCodec`):
  ratio 9.5, ~6.5 ms encode / 0.35 ms decode per timestep, full archive
  ≈ 191 GB (`scripts/benchmark_compression.py`, 360 timesteps over 5 years
  and all seasons). On uint8, Blosc byte shuffle is a no-op and bitshuffle
  is 21–27 % larger and decodes 2.5× slower, so don't use shuffling.
  zstd 19 is ~5 % smaller at ~16× encode cost. The validator warns
  "zstdcodec, recommended is zstd" because it reads v3 codec class names;
  that warning is harmless.
- The validator treats any gap as a variable timestep, so the global attr
  `consistent_timestep_start` is always set (to the first timestep).
- Data must decode to float. We store uint8 with `scale_factor=0.5`,
  `add_offset=-32`, `_FillValue=255`; xarray decodes to float64 (Zarr v3
  JSON attrs lose the float32 type), which the spec allows.
- Variable `dbz`, `standard_name=equivalent_reflectivity_factor`,
  `units=dBZ`, `grid_mapping=spatial_ref`.
- `spatial_ref` carries CF grid-mapping attrs plus `crs_wkt`/`spatial_ref`
  WKT with BBOX. 2D `lat`/`lon` coordinates are required.
- Global attrs: SPDX `license`, `mlcast_created_on/by/with`,
  `mlcast_dataset_version`, `mlcast_dataset_validator_version`,
  `mlcast_dataset_identifier` (`FI-FMI-dbz`). `mlcast_created_with` must
  point to an existing tagged GitHub revision.

## Conventions

- Python, dependencies via `uv` (`uv sync`, `uv run ...`). `uv` uses its own
  managed CPython (`python-preference = "only-managed"`): the site Python
  under `/tutka/code/dev` lacks `_bz2`, which numcodecs needs.
- The dataset license differs from the code license (repo `LICENSE` is for
  code only).
- Never modify or write anything under `/arch/radar/storage`; it is the
  operational archive. Read-only.
- Write outputs and temporary files outside the repo (or to gitignored
  paths); don't commit large data.
- `uv run ruff check` and `uv run ruff format` must pass; `uv run pytest`
  must pass.
