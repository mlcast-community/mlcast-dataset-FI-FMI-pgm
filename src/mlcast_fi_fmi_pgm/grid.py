"""Grid geometry and CF georeferencing for the SUOMI1 composite."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from pyproj import CRS, Transformer

from . import config


@dataclass(frozen=True)
class Grid:
    proj_string: str = config.PROJ_STRING
    width: int = config.WIDTH
    height: int = config.HEIGHT
    bottom_left: tuple[float, float] = config.BOTTOM_LEFT_LONLAT
    top_right: tuple[float, float] = config.TOP_RIGHT_LONLAT

    @cached_property
    def crs(self) -> CRS:
        return CRS.from_proj4(self.proj_string)

    @cached_property
    def _edges(self) -> tuple[float, float, float, float]:
        """Projected outer edges (x_min, y_min, x_max, y_max)."""
        to_xy = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        x0, y0 = to_xy.transform(*self.bottom_left)
        x1, y1 = to_xy.transform(*self.top_right)
        return x0, y0, x1, y1

    @property
    def dx(self) -> float:
        x0, _, x1, _ = self._edges
        return (x1 - x0) / self.width

    @property
    def dy(self) -> float:
        _, y0, _, y1 = self._edges
        return (y1 - y0) / self.height

    @property
    def x(self) -> np.ndarray:
        """Pixel-centre x coordinates (m), west to east."""
        return self._edges[0] + (np.arange(self.width) + 0.5) * self.dx

    @property
    def y(self) -> np.ndarray:
        """Pixel-centre y coordinates (m), north to south (PGM row order)."""
        return self._edges[3] - (np.arange(self.height) + 0.5) * self.dy

    @property
    def geotransform(self) -> tuple[float, float, float, float, float, float]:
        x0, _, _, y1 = self._edges
        return (x0, self.dx, 0.0, y1, 0.0, -self.dy)

    def latlon(self) -> tuple[np.ndarray, np.ndarray]:
        """2D (lat, lon) arrays of pixel centres, shape (height, width)."""
        to_ll = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        xx, yy = np.meshgrid(self.x, self.y)
        lon, lat = to_ll.transform(xx, yy)
        return lat, lon

    def spatial_ref_attrs(self, lat: np.ndarray, lon: np.ndarray) -> dict:
        """CF grid-mapping attributes plus WKT (with BBOX) for the CRS variable."""
        wkt = self.crs.to_wkt()
        if "BBOX[" not in wkt:
            bbox = f"BBOX[{lat.min():.6f},{lon.min():.6f},{lat.max():.6f},{lon.max():.6f}]"
            wkt = wkt[:-1] + f",{bbox}]"
        cf = {k: v for k, v in self.crs.to_cf().items() if k != "crs_wkt"}
        return {
            **cf,
            "crs_wkt": wkt,
            "spatial_ref": wkt,
            "proj4": self.proj_string,
            "GeoTransform": " ".join(repr(float(v)) for v in self.geotransform),
        }
