# FMI radar composite to Zarr

This repository contains the code used to build a Zarr v3 version of the
Finnish Meteorological Institute (FMI) radar reflectivity composite for the
[MLCast](https://github.com/mlcast-community/mlcast-datasets) dataset
collection ([issue #64](https://github.com/mlcast-community/mlcast-datasets/issues/64)).

## Important note on licensing

The license of this software is **not** the same as the license of the dataset itself.

The FMI radar composite dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Please refer to the repository `LICENSE` file for the license that applies to this codebase.

## Dataset overview

| | |
| --- | --- |
| Product | FMI composite of the lowest radar sweeps (`lowest` CAPPI), `FIN_SUOMI1` area |
| Variable | `dbz`: equivalent reflectivity factor (dBZ), from the `CorrectedReflectivity` field |
| Timestep | 5 minutes |
| Period | 2008-03-18 05:00 UTC to 2026-11-02 |
| Completeness | about 99 % of 5-minute timesteps; gaps are listed in `missing_times` |
| Grid | 760 × 1226 pixels, about 1 km, polar stereographic (`lon_0=25`, `lat_ts=60`, spherical earth R = 6371288 m) |
| Extent | bottom-left 18.600° E 57.930° N, top-right 34.903° E 69.005° N (outer pixel edges) |
| Size | ≈ 1.94 million timesteps, ≈ 190 GB compressed |

The dataset follows the MLCast `source_data/radar_precipitation` specification
(validated with [`mlcast-dataset-validator`](https://github.com/mlcast-community/mlcast-dataset-validator) 0.4.0):

- **Dimensions:** `dbz(time, y, x)`. `y` runs north to south.
- **Coordinates:** projected `x`/`y` (pixel centres, in metres), 2D `lat`/`lon`, and `spatial_ref`, which carries the CF grid-mapping attributes and a WKT with a bounding box.
- **Time:** `time` contains only timesteps with data. Missing 5-minute timesteps are listed in `missing_times`.
- **Radar network:** the coordinate `radar_status(time, radar)` tells which radars contributed to each timestep: 1 = contributed, 0 = part of the network but not contributing (e.g. during maintenance), −1 = not listed. `radar_lon`/`radar_lat` give the radar positions.
- **Storage:** values are stored as `uint8` with CF packing: `dBZ = 0.5 * value - 32`, and 255 means outside radar coverage (decoded as NaN). xarray decodes them to floating-point dBZ. Raw value 0 (−32 dBZ) means no detected echo.

## Project overview

The workflow in this repository consists of five main steps:

1. **Build an inventory parquet file**
2. **Verify every source file**
3. **Create an empty Zarr dataset**
4. **Populate the Zarr dataset with data from the source PGM files**
5. **Add the radar network status from the file headers**

The final dataset is stored in **Zarr v3** format.

## Workflow

### 1. Build the inventory

The source data are gzip-compressed binary PGM (`P5`) files, one per
timestep, in FMI's internal radar archive (root directory set by
`ARCHIVE_ROOT` in `config.py`):

```text
<archive root>/{year}/{month:02d}/{day:02d}/fmi/radar/composite/lowest/
    {year}{month:02d}{day:02d}{hour:02d}{minute:02d}_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz
```

The inventory step scans these directories and writes a parquet file with
the timestamp, path and size of every file. Two quirks of the archive are
handled here:

- **Directory dates** do not always match file timestamps. For example
  `2015/06/01/` holds files from `201505312340` to `201506012335`. Timestamps
  are always taken from the filename, and neighbouring day directories are
  scanned too. Copies of the same file found in two directories are
  identical, and only one is kept.
- **Compression:** some files are stored uncompressed (`.pgm` instead of
  `.pgm.gz`): all of January–September 2023, everything from 2026-09-19 on,
  and scattered files in other years. Both forms are read.

The inventory is **not included** in this repository, because building it
requires access to FMI's internal archive.

### 2. Verify the source files

A small number of archived files are corrupt: about 820 of 1.94 million,
mostly from 2008–2011. Examples are files of zero bytes and broken gzip
streams. The verify step reads every file once, checks its header (size,
maxval, grid corners, and that the observation time matches the filename),
and writes `inventory.bad.csv`. Those timesteps are then treated as missing,
so they appear in `missing_times` rather than as empty fields in `time`.

### 3. Create the empty Zarr dataset

The empty store holds the coordinates, the CRS and all metadata; the data
array itself is not yet written. The grid is defined in
`src/mlcast_fi_fmi_pgm/config.py` and `grid.py`. Its geometry was checked
against the data: the corner coordinates in the PGM header are the outer
pixel edges, and row 0 is the northern edge.

The data variable is stored with:

- **Chunks:** one per timestep (`1 × 1226 × 760`), as the MLCast spec requires.
- **Shards:** 2016 timesteps (one week) per shard file, which keeps the file count low on HPC filesystems.
- **Compression:** plain **zstd, level 9**. See [Compression settings](#compression-settings).

A `<store>.sources.parquet` file next to the store records the source file
behind every time index.

### 4. Populate the Zarr dataset

Each shard is read and written in full by a single worker process (with
reader threads inside), so no locking is needed. Finished shards are
recorded in `<store>.progress/`, so an interrupted run resumes where it
stopped when you rerun the same command. Before reading and before writing
each shard, the remaining free disk space is checked. If it would drop below
max(500 GB, 10 % of the filesystem), the run stops cleanly, without writing
partial data.

### 5. Add the radar network status

Every PGM header lists the radars of the composite, one line each:
`# radar <code> <flag> <lon> <lat>`. This step reads only the headers (the
first few KB of each file), saves them to `<store>.radars.parquet`, and adds
`radar_status(time, radar)`, `radar_lon` and `radar_lat` to the store
without touching the reflectivity data. It can run any time after step 3.

The flag was checked against the data: when a radar has flag 0, its own
coverage area turns into nodata compared with a neighbouring timestep where
it has flag 1, and nothing else changes. Between files where every radar has
the same status, no coverage is lost anywhere.

`radar_status` is stored as a coordinate rather than a data variable,
because the MLCast specification expects every data variable to be a
gridded `(time, y, x)` field.

```python
(ds.radar_status == 1).sum("radar")  # number of contributing radars per timestep
ds.radar_status.sel(radar="KOR")  # history of one radar
```

## Repository structure

The Python package is located in `src/mlcast_fi_fmi_pgm/`:

- `config.py`: archive paths, grid constants, value scaling and output defaults
- `pgm.py`: PGM reader and header validation
- `grid.py`: projected and geographic coordinates, CF grid mapping and WKT
- `inventory.py`: archive scan and construction of the time axis and missing times
- `verify.py`: reads every file once and lists the unusable ones
- `create.py`: creates the empty Zarr v3 dataset
- `populate.py`: writes the data, one shard per task
- `radars.py`: reads the radar lists from the file headers and adds `radar_status`
- `main.py`: command-line entry point (`fmi-pgm-zarr`)

Other files:

- `tests/`: pytest suite, with one sample PGM file in `tests/data/`
- `scripts/benchmark_compression.py`: compression codec benchmark
- `AGENTS.md`: detailed design notes and findings about the source data

## Installation

All dependencies can be installed with [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
```

uv is configured to use its own managed CPython 3.12
(`python-preference = "only-managed"`).

## Usage

The workflow is run with the `fmi-pgm-zarr` command, one subcommand per step:

```bash
uv run fmi-pgm-zarr inventory --start 2008-01-01 --end 2026-11-02
uv run fmi-pgm-zarr verify
uv run fmi-pgm-zarr create
uv run fmi-pgm-zarr populate --workers 6 --threads 8
uv run fmi-pgm-zarr radars --workers 6 --threads 16
```

With 6 workers a full build takes about 2 hours.

Main options:

| Subcommand | Flag | Default | Description |
| --- | --- | --- | --- |
| all | `--inventory` | `OUTPUT_DIR/inventory.parquet` | Inventory parquet file. |
| `inventory` | `--start`, `--end` | `2008-01-01`, today | Time range to scan, both ends inclusive. |
| `inventory` | `--root` | `ARCHIVE_ROOT` in `config.py` | Archive root directory. |
| `verify`, `populate`, `radars` | `--workers` | `6` / `4` / `6` | Parallel processes. |
| `verify`, `populate`, `radars` | `--threads` | `8` | File reader threads per process. |
| `create`, `populate`, `radars` | `--store` | `OUTPUT_DIR/fi-fmi-dbz-5min.zarr` | Output Zarr store. |
| `create` | `--start`, `--end` | full inventory | Time range of the store, both ends inclusive. |
| `create` | `--shard-size` | `2016` | Timesteps per shard. |
| `create` | `--clevel` | `9` | zstd compression level. |
| `create` | `--license`, `--created-by`, `--dataset-version` | see `config.py` | Global metadata attributes. |
| `create` | `--overwrite` | off | Replace an existing store. |
| `create` | `--allow-unverified` | off | Create the store without running `verify` first (not recommended). |
| `populate` | `--redo` | off | Rewrite shards that are already marked done. |
| `populate` | `--shards` | all | Write only the listed shard indices. |
| `radars` | `--rescan` | off | Reread the headers even if `<store>.radars.parquet` exists. |
| `populate` | `--min-free-gb` | max(500 GB, 10 % of filesystem) | Free-space threshold below which the run stops. |

Run `uv run fmi-pgm-zarr <subcommand> --help` for the full, up-to-date list.

### Testing on a subset of the dataset

To iterate quickly, build a store for a short period with `--start`/`--end`.
These options select from the inventory, which must already exist and have
been verified:

```bash
uv run fmi-pgm-zarr create --start 2024-06-01 --end "2024-06-07 23:55" \
    --store ./fi-fmi-dbz-test.zarr --overwrite
uv run fmi-pgm-zarr populate --store ./fi-fmi-dbz-test.zarr --workers 1 --threads 16
```

One week (2016 timesteps, one shard) takes well under a minute and about 240 MB.

### Validation

```bash
uv run mlcast.validate_dataset source_data radar_precipitation <store>
```

The validator's GDAL checks need GDAL's Python bindings (`osgeo`), which are
not installed by `uv sync`. Run the validator additionally in an environment
that has `gdal`, `rioxarray`, `zarr` and `mlcast-dataset-validator` installed
(e.g. from conda-forge) to cover them.

Expected warnings:

- **Compression name:** the validator reports `zstdcodec, recommended is zstd`. The data is zstd-compressed; the validator reads the name of the Zarr v3 codec class rather than the algorithm.
- **Creator software GitHub revision:** this warning appears until the tag referenced in `mlcast_created_with` exists on GitHub.

### Compression settings

Codecs were compared on 360 timesteps sampled across all months of 2010,
2014, 2018, 2022 and 2025 (`scripts/benchmark_compression.py`), with
single-threaded timings per timestep:

| Codec | Ratio | Encode ms | Decode ms | Full archive |
| --- | --- | --- | --- | --- |
| zstd 3 | 9.16 | 1.8 | 0.36 | 197 GB |
| **zstd 9 (used)** | **9.48** | **6.5** | **0.35** | **191 GB** |
| zstd 19 | 9.96 | 107 | 0.39 | 182 GB |
| Blosc-zstd 5, no shuffle or byte shuffle | 9.47 | 6.5 | 0.38 | 191 GB |
| Blosc-zstd 5, bitshuffle | 7.48 | 9.5 | 0.88 | 242 GB |
| Blosc-zstd 9, no shuffle or byte shuffle | 10.07 | 1008 | 0.38 | 179 GB |
| Blosc-zstd 9, bitshuffle | 8.33 | 309 | 0.88 | 217 GB |

Because the data are stored as `uint8`, byte shuffle has no effect (it gives
output identical to no shuffle). Bitshuffle makes the output 21–27 % larger
in every season and decodes 2.5× slower, because it breaks up the long runs
of 0 (no echo) and 255 (no coverage) that zstd compresses well. Shuffling
pays off for multi-byte types such as float32, but not here.

To rerun the benchmark:

```bash
uv run python scripts/benchmark_compression.py --years 2010 2014 2018 2022 2025 --per-month 6
```

### Development

```bash
uv run pytest
uv run ruff check
uv run ruff format
```

## Notes

The inventory, bad-file list and source paths are environment-specific and
depend on access to FMI's internal radar archive. The output dataset uses
Zarr v3 with sharding.

## Authors

- Jussi Tiira (jussi.tiira@fmi.fi), contact person (`mlcast_created_by`)
- Joonas Karjalainen (joonas.karjalainen@fmi.fi)
- Jenna Ritvanen (jenna.ritvanen@fmi.fi)

Finnish Meteorological Institute. All three are listed in the dataset's
`creator_name` and `creator_email` attributes.

## Dataset license

The FMI radar composite dataset is licensed under CC BY 4.0.
