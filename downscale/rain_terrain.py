"""Static terrain covariates aligned to the rainfall target grid."""
from __future__ import annotations

import hashlib
import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from scipy.spatial import Delaunay
from pyproj import Transformer

from .constants import G

CHANNELS = ["elevation_km", "fine_minus_coarse_km", "slope_east_percent",
            "slope_north_percent", "relief_500m", "land_fraction",
            "coast_proximity", "fine_coverage"]


def make_terrain(fine, coarse, lat, lon):
    """Average static fields over nine subcell samples, then derive target-scale slopes.

    Fine input has 2D lat/lon, elevation in metres and a fractional land mask.
    Coarse input has ascending 1D lat/lon, elevation in metres and land fraction.
    Fine-crop gaps fall back to coarse terrain with an explicit coverage channel.
    """
    lat, lon = np.asarray(lat), np.asarray(lon)
    if min(len(lat), len(lon)) < 3 or not np.all(np.diff(lat) > 0) or not np.all(np.diff(lon) > 0):
        raise ValueError("terrain target needs ascending lat/lon with at least three cells")
    if fine.elevation.attrs.get("units") != "m" or coarse.elevation.attrs.get("units") != "m":
        raise ValueError("terrain elevation must be in metres")
    if fine.lat.ndim != 2 or fine.lon.shape != fine.lat.shape:
        raise ValueError("fine terrain needs matching 2D lat/lon")
    for ds in (fine, coarse):
        if not np.isfinite(ds.elevation.values).all() or not np.isfinite(ds.land_fraction.values).all():
            raise ValueError("terrain source has nonfinite values")
        if np.any((ds.elevation.values < -500) | (ds.elevation.values > 10000)):
            raise ValueError("terrain elevation outside plausible metre range")
        if np.any((ds.land_fraction.values < 0) | (ds.land_fraction.values > 1)):
            raise ValueError("land fractions must lie between zero and one")
    dy, dx = np.median(np.diff(lat)), np.median(np.diff(lon))
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    sample_lat = np.stack([yy + a * dy / 3 for a in (-1, 0, 1) for _ in (-1, 0, 1)])
    sample_lon = np.stack([xx + b * dx / 3 for _ in (-1, 0, 1) for b in (-1, 0, 1)])
    points = np.column_stack([sample_lat.ravel(), sample_lon.ravel()])
    coarse_height = RegularGridInterpolator((coarse.lat.values, coarse.lon.values), coarse.elevation.values)(points).reshape(sample_lat.shape)
    coarse_land = RegularGridInterpolator((coarse.lat.values, coarse.lon.values), coarse.land_fraction.values)(points).reshape(sample_lat.shape)
    # Equal-area European coordinates avoid triangulating in distorted degrees.
    project = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    fx, fy = project.transform(fine.lon.values.ravel(), fine.lat.values.ravel())
    qx, qy = project.transform(sample_lon.ravel(), sample_lat.ravel())
    triangulation = Delaunay(np.column_stack([fx, fy]))
    query = np.column_stack([qx, qy])
    height = LinearNDInterpolator(triangulation, fine.elevation.values.ravel())(query).reshape(sample_lat.shape)
    land = LinearNDInterpolator(triangulation, fine.land_fraction.values.ravel())(query).reshape(sample_lat.shape)
    covered = np.isfinite(height) & np.isfinite(land)
    height = np.where(covered, height, coarse_height)
    land = np.where(covered, land, coarse_land)
    weights = np.cos(np.deg2rad(sample_lat))
    weights /= weights.sum(0, keepdims=True)
    elevation = (height * weights).sum(0)
    relief = np.sqrt(np.maximum((height ** 2 * weights).sum(0) - elevation ** 2, 0))
    land_fraction = np.clip((land * weights).sum(0), 0, 1)
    radius = 6371000.0
    north = np.gradient(elevation, np.deg2rad(lat) * radius, axis=0)
    east = np.gradient(elevation, np.deg2rad(lon), axis=1) / (radius * np.cos(np.deg2rad(lat))[:, None])
    distance = distance_transform_edt(land_fraction > .5,
                                    sampling=(np.deg2rad(dy) * radius / 1000,
                                              np.deg2rad(dx) * radius * np.cos(np.deg2rad(lat.mean())) / 1000))
    coast_proximity = np.exp(-distance / 25) if np.any(land_fraction <= .5) else np.zeros_like(distance)
    features = np.stack([elevation / 1000, (elevation - (coarse_height * weights).sum(0)) / 1000,
                         east * 100, north * 100, relief / 500, land_fraction,
                         coast_proximity, (covered * weights).sum(0)]).astype("float32")
    result = xr.Dataset({"static": (("static_channel", "lat", "lon"), features),
                         "elevation": (("lat", "lon"), elevation.astype("float32")),
                         "land_fraction": (("lat", "lon"), land_fraction.astype("float32"))},
                        coords={"static_channel": CHANNELS, "lat": lat, "lon": lon})
    result.elevation.attrs["units"] = "m"
    result.attrs = {"source": "CERRA 5.5 km terrain, with ERA5 static fallback outside the cached fine crop",
                    "aggregation": "3x3 target-cell midpoint quadrature weighted by cos(latitude)",
                    "slopes": "east/north gradients at target resolution; percent rise",
                    "coast": "exp(-distance_to_sea_km/25); distance uses mean target-grid spacing",
                    "coverage_mean": float(features[-1].mean()),
                    "sha256": hashlib.sha256(features.tobytes()).hexdigest()}
    return result


def build_terrain(lat, lon, fine_path):
    """Build once from a cached CERRA static pack and two public ERA5 static fields."""
    from .precip_data import crop, open_store
    from .sources import ERA5_ARCO
    fine = xr.open_zarr(fine_path).load()
    field = "orography" if "orography" in fine else "cerra_orography"
    fine = xr.Dataset({"elevation": fine[field], "land_fraction": fine.land_sea_mask},
                      coords={"lat": fine.lat, "lon": fine.lon})
    source = open_store(ERA5_ARCO)[["geopotential_at_surface", "land_sea_mask"]]
    source = crop(source.sel(time="2000-01-01T00:00", drop=True),
                  (float(np.min(lat)) - 1, float(np.max(lat)) + 1, float(np.min(lon)) - 1, float(np.max(lon)) + 1)).load()
    coarse = xr.Dataset({"elevation": source.geopotential_at_surface / G, "land_fraction": source.land_sea_mask})
    coarse.elevation.attrs["units"] = "m"
    return make_terrain(fine, coarse, lat, lon)


def load_static(path, lat, lon):
    ds = xr.open_zarr(path).load()
    for name, values in (("lat", lat), ("lon", lon)):
        if not np.array_equal(ds[name], values):
            raise ValueError(f"static terrain {name} differs from the rainfall grid")
    if ds.static_channel.values.tolist() != CHANNELS or not np.isfinite(ds.static.values).all():
        raise ValueError("invalid static terrain channels or values")
    return ds.static.values.astype("float32"), dict(ds.attrs)


def expand_terrain_weights(state, target_state, count):
    """Insert zero-weight terrain channels before the existing noise channel."""
    key = "unet.inc.net.0.weight"
    state = dict(state)
    old, new = state[key], target_state[key]
    if old.shape == new.shape:
        return state
    if old.shape[0] != new.shape[0] or old.shape[2:] != new.shape[2:] or new.shape[1] - old.shape[1] != count:
        raise ValueError("warm-start input channels are incompatible with static terrain")
    expanded = new.detach().clone().zero_()
    expanded[:, :old.shape[1] - 1] = old[:, :-1]
    expanded[:, -1:] = old[:, -1:]
    state[key] = expanded
    return state


def terrain_scores(prediction, pairs, static):
    """Pointwise rainfall scores by land-elevation band on the same target grid."""
    from .precipitation import score
    elevation, land = static[0] * 1000, static[5] > .5
    result = {}
    for name, mask in {"lowland_below_200m": land & (elevation < 200),
                       "hills_200_to_1000m": land & (elevation >= 200) & (elevation < 1000),
                       "mountains_above_1000m": land & (elevation >= 1000)}.items():
        if not mask.any():
            continue
        truth = pairs.target.values[:, mask, None]
        pred = prediction.precipitation.values[:, :, mask, None]
        coarse = pairs.predictors.sel(variable="tp").values[:, None, mask, None]
        result[name] = {"cells": int(mask.sum()), "model": score(pred, truth), "baseline": score(coarse, truth)}
    return result
