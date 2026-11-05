import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from mlcast_fi_fmi_pgm import config
from mlcast_fi_fmi_pgm.create import create_store
from mlcast_fi_fmi_pgm.grid import Grid
from mlcast_fi_fmi_pgm.inventory import build_time_axis
from mlcast_fi_fmi_pgm.pgm import PGMError, check_header, read_pgm, read_pgm_header
from mlcast_fi_fmi_pgm.populate import populate_store
from mlcast_fi_fmi_pgm.radars import add_radars, parse_radar_lines

DATA = Path(__file__).parent / "data"
SAMPLE = DATA / "202406011200_fmi.radar.composite.lowest_FIN_SUOMI1.pgm.gz"
SAMPLE_TIME = pd.Timestamp("2024-06-01 12:00")


def test_read_sample():
    header, data = read_pgm(SAMPLE)
    assert data.shape == (config.HEIGHT, config.WIDTH)
    assert data.dtype == np.uint8
    assert header.obstime == SAMPLE_TIME
    assert header.meta["param"] == "CorrectedReflectivity"
    assert len(header.radars) == 11
    check_header(header, SAMPLE_TIME)
    with pytest.raises(PGMError):
        check_header(header, SAMPLE_TIME + pd.Timedelta("5min"))


def test_grid_geometry():
    g = Grid()
    assert g.dx == pytest.approx(999.674053, abs=1e-3)
    assert g.dy == pytest.approx(999.62859, abs=1e-3)
    assert g.geotransform[0] == pytest.approx(0, abs=0.1)
    assert g.y[0] > g.y[-1]  # north-up
    lat, lon = g.latlon()
    assert lat.shape == (config.HEIGHT, config.WIDTH)
    assert 68.5 < lat[0].max() < 70.5 and 57.5 < lat[-1].min() < 58.5


def test_time_axis():
    t = pd.to_datetime(
        [
            "2024-01-01 00:00",
            "2024-01-01 00:05",
            "2024-01-01 00:05",
            "2024-01-01 00:12",
            "2024-01-01 00:20",
            "2024-01-01 00:25",
        ]
    ).astype("datetime64[ns]")
    inv = pd.DataFrame({"time": t, "path": list("abcdef"), "size": [1, 1, 1, 1, 0, 1]})
    axis = build_time_axis(inv)
    assert list(axis.files["path"]) == ["a", "b", "f"]
    missing = pd.DatetimeIndex(axis.missing_times).strftime("%H:%M")
    assert list(missing) == ["00:10", "00:15", "00:20"]
    assert sorted(axis.dropped["reason"]) == [
        "duplicate timestamp",
        "empty or unreadable file",
        "off timestep grid",
    ]


def _copy_with_time(t: pd.Timestamp, dest: Path, replace: dict[bytes, bytes] | None = None) -> Path:
    raw = (
        gzip.open(SAMPLE)
        .read()
        .replace(
            b"# obstime " + SAMPLE_TIME.strftime("%Y%m%d%H%M").encode(),
            b"# obstime " + t.strftime("%Y%m%d%H%M").encode(),
        )
    )
    for old, new in (replace or {}).items():
        assert old in raw
        raw = raw.replace(old, new)
    path = dest / config.FILENAME_TEMPLATE.format(
        year=t.year, month=t.month, day=t.day, hour=t.hour, minute=t.minute
    )
    path.write_bytes(gzip.compress(raw))
    return path


def test_end_to_end(tmp_path):
    times = pd.to_datetime(["2024-06-01 12:00", "2024-06-01 12:05", "2024-06-01 12:15"])
    paths = [_copy_with_time(t, tmp_path) for t in times]
    paths[1].write_bytes(b"garbage")  # unreadable file → fill, reported
    inv = pd.DataFrame({"time": times.astype("datetime64[ns]"), "path": map(str, paths), "size": 1})
    axis = build_time_axis(inv)

    store = tmp_path / "out.zarr"
    create_store(store, axis, shard_size=2)
    assert zarr.open_array(store, path="dbz").fill_value == config.NODATA
    failures = populate_store(store, workers=1, threads=2, min_free=0)
    assert len(failures) == 1

    ds = xr.open_zarr(store)
    # JSON attrs make scale_factor float64, so xarray decodes to float64 (allowed by spec)
    assert np.issubdtype(ds["dbz"].dtype, np.floating)
    assert ds["dbz"].encoding["dtype"] == np.uint8
    assert list(ds["missing_times"].dt.strftime("%H:%M").values) == ["12:10"]
    assert ds["dbz"].encoding["chunks"] == (1, config.HEIGHT, config.WIDTH)

    _, raw = read_pgm(SAMPLE)
    expected = np.where(raw == 255, np.nan, raw * 0.5 - 32)
    np.testing.assert_array_equal(ds["dbz"].isel(time=0).values, expected)
    np.testing.assert_array_equal(ds["dbz"].isel(time=2).values, expected)
    assert np.isnan(ds["dbz"].isel(time=1).values).all()

    # restart: nothing left to do
    assert len(populate_store(store, workers=1, threads=2, min_free=0)) == 1


def test_filename_pattern():
    base = "202301011200_fmi.radar.composite.lowest_FIN_SUOMI1"
    assert config.FILENAME_RE.match(base + ".pgm.gz").group(1) == "202301011200"
    assert config.FILENAME_RE.match(base + ".pgm")
    for other in (base + ".png", base.replace("SUOMI1", "SUOMI1BIG") + ".pgm"):
        assert not config.FILENAME_RE.match(other)


def test_time_axis_bad_paths():
    t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 00:05"])
    inv = pd.DataFrame({"time": t.astype("datetime64[ns]"), "path": ["a", "b1", "b2"], "size": 1})
    axis = build_time_axis(inv, bad_paths={"b1"})  # good duplicate replaces the bad one
    assert list(axis.files["path"]) == ["a", "b2"]
    axis = build_time_axis(inv, bad_paths={"b1", "b2"})  # no usable copy → missing
    assert list(axis.files["path"]) == ["a"]
    assert len(axis.missing_times) == 0  # trailing gap is outside the series
    axis = build_time_axis(
        pd.concat([inv, inv.iloc[:1].assign(time=pd.Timestamp("2024-01-01 00:10"), path="c")]),
        bad_paths={"b1", "b2"},
    )
    assert list(pd.DatetimeIndex(axis.missing_times).strftime("%H:%M")) == ["00:05"]


@pytest.mark.parametrize("damage", ["zeros", "corrupt_stream", "truncated"])
def test_damaged_files_raise_pgmerror(tmp_path, damage):
    good = SAMPLE.read_bytes()
    bad = {
        "zeros": b"\0" * 1000,
        "corrupt_stream": good[:20] + bytes(b ^ 0xFF for b in good[20:2000]) + good[2000:],
        "truncated": good[: len(good) // 2],
    }[damage]
    path = tmp_path / SAMPLE.name
    path.write_bytes(bad)
    with pytest.raises(PGMError):
        read_pgm(path)


def test_populate_stops_on_low_disk_space(tmp_path):
    from mlcast_fi_fmi_pgm.populate import DiskSpaceError, default_min_free, progress_dir

    t = pd.Timestamp("2024-06-01 12:00")
    path = _copy_with_time(t, tmp_path)
    inv = pd.DataFrame({"time": [t], "path": [str(path)], "size": 1})
    inv["time"] = inv["time"].astype("datetime64[ns]")
    store = tmp_path / "out.zarr"
    create_store(store, build_time_axis(inv), shard_size=1)

    assert default_min_free(store) >= config.MIN_FREE_BYTES
    impossible = 10**18  # 1 EB
    with pytest.raises(DiskSpaceError):
        populate_store(store, workers=1, threads=1, min_free=impossible)
    assert not list(progress_dir(store).glob("*.done"))  # nothing marked done
    assert np.isnan(xr.open_zarr(store)["dbz"].values).all()  # nothing written

    populate_store(store, workers=1, threads=1, min_free=0)  # resumes normally
    assert len(list(progress_dir(store).glob("*.done"))) == 1


def test_read_header_only(tmp_path):
    full, _ = read_pgm(SAMPLE)
    assert read_pgm_header(SAMPLE) == full
    assert read_pgm_header(SAMPLE, nbytes=64) == full  # header longer than nbytes
    plain = tmp_path / SAMPLE.name.removesuffix(".gz")
    plain.write_bytes(gzip.open(SAMPLE).read())
    assert read_pgm_header(plain) == full
    plain.write_bytes(b"P5\n# radar KOR 1 21.6 60.1\n760 1226 2")  # cut inside maxval
    with pytest.raises(PGMError):
        read_pgm_header(plain)


def test_parse_radar_lines():
    assert parse_radar_lines(["KOR 1 21.6460 60.1280"]) == [("KOR", 1, 21.646, 60.128)]
    for bad in ("KOR 1 21.6", "KOR 2 21.6 60.1"):
        with pytest.raises(PGMError):
            parse_radar_lines([bad])


def test_radar_status(tmp_path):
    times = pd.to_datetime(["2024-06-01 12:00", "2024-06-01 12:05", "2024-06-01 12:15"])
    paths = [
        _copy_with_time(times[0], tmp_path),
        _copy_with_time(times[1], tmp_path, {b"# radar PET 1 ": b"# radar PET 0 "}),
        _copy_with_time(times[2], tmp_path, {b"# radar KOR 1 21.6460 60.1280\n": b""}),
    ]
    inv = pd.DataFrame({"time": times.astype("datetime64[ns]"), "path": map(str, paths), "size": 1})
    store = tmp_path / "out.zarr"
    create_store(store, build_time_axis(inv), shard_size=2)
    populate_store(store, workers=1, threads=2, min_free=0)
    before = xr.open_zarr(store)
    attrs, dbz = dict(before.attrs), before["dbz"].values

    for _ in range(2):  # a rerun replaces the variables
        add_radars(store, workers=1, threads=2)
    ds = xr.open_zarr(store)
    assert list(ds.data_vars) == ["dbz"]  # spec rules apply only to the radar field
    assert ds.attrs == attrs
    np.testing.assert_array_equal(ds["dbz"].values, dbz)

    status = ds["radar_status"]
    assert status.dtype == np.int8
    assert status.sizes == {"time": 3, "radar": 11}
    assert list(ds["radar"].values) == sorted(ds["radar"].values)
    assert status.sel(radar="KOR").values.tolist() == [1, 1, -1]
    assert status.sel(radar="PET").values.tolist() == [1, 0, 1]
    assert (status.sel(radar="ANJ") == 1).all()
    assert float(ds["radar_lon"].sel(radar="KOR")) == pytest.approx(21.646)
    assert float(ds["radar_lat"].sel(radar="KOR")) == pytest.approx(60.128)
    assert zarr.open_array(store, path="radar_status").fill_value == -1
