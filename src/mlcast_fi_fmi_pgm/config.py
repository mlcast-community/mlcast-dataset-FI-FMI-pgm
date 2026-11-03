"""Constants describing the FMI SUOMI1 composite and the default dataset settings."""

from __future__ import annotations

import re

# --- Source archive ---------------------------------------------------------

ARCHIVE_ROOT = "/arch/radar/storage"
DAY_DIR_TEMPLATE = "{root}/{year}/{month:02d}/{day:02d}/fmi/radar/composite/lowest"
FILENAME_TEMPLATE = (
    "{year}{month:02d}{day:02d}{hour:02d}{minute:02d}_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz"
)
# Mostly gzipped, but 2023-01..09 and from 2026-09-19 on the files are plain .pgm
FILENAME_RE = re.compile(r"^(\d{12})_fmi\.radar\.composite\.lowest_FIN_SUOMI1\.pgm(\.gz)?$")

TIMESTEP = "5min"

# --- Grid -------------------------------------------------------------------

WIDTH = 760
HEIGHT = 1226
MAXVAL = 255

PROJ_STRING = (
    "+proj=stere +a=6371288 +lon_0=25E +lat_0=90N +lat_ts=60 "
    "+x_0=380886.310 +y_0=3395677.920 +no_defs"
)
# Outer edges of the grid (lon, lat). Verified: these project to (0, 0) and
# (WIDTH * 999.674 m, HEIGHT * 999.629 m), i.e. pixel *edges*, not centres.
BOTTOM_LEFT_LONLAT = (18.6, 57.93)
TOP_RIGHT_LONLAT = (34.903, 69.005)

# --- Encoding of pixel values ----------------------------------------------

NODATA = 255  # outside radar coverage
GAIN = 0.5  # dBZ = GAIN * value + OFFSET
OFFSET = -32.0

# --- Output defaults -------------------------------------------------------

OUTPUT_DIR = "/tutka/data/storage/mlcast-dataset-fi-fmi-pgm"
VARIABLE_NAME = "dbz"
SHARD_SIZE = 2016  # timesteps per shard (one week at 5 min)
CLEVEL = 9  # zstd level

DATASET_VERSION = "0.1.0"
VALIDATOR_VERSION = "0.4.0"
REPO_URL = "https://github.com/mlcast-community/mlcast-dataset-FI-FMI-pgm"
LICENSE = "CC-BY-4.0"
CREATED_BY = "Jussi Tiira <jussi.tiira@fmi.fi>"
