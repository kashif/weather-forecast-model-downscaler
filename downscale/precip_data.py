"""Hourly rainfall training data and Earth2Studio-compatible observation sources.

Rain depths are in metres at source interfaces, and millimetres in training packs.
Every timestamp labels the END of its accumulation interval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .constants import REGION_PRESETS
from .regions import crop_indices, to_signed_lon
from .sources import ERA5_ARCO

PARDIG_NRT = "gs://pardig_1/NRT/zarr/2016_to_2025/predictions.zarr"
SATWEAVE = "gs://satweave_1/zarr/2016_to_2025/mosaic.zarr"
PARDIG_TIMESERIES = PARDIG_NRT.replace("/zarr/", "/zarr_rechunked_as_timeseries/")
ERA5_NAMES = {
    "tp": "total_precipitation", "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature", "u10m": "10m_u_component_of_wind",
    "v10m": "10m_v_component_of_wind", "msl": "mean_sea_level_pressure",
}


def open_store(url: str) -> xr.Dataset:
    # Backend-lazy indexing avoids constructing a Dask task for every hour in
    # the global, multi-decade ERA5 archive before narrowing the request.
    if url.startswith("gs://"):
        import obstore
        import zarr
        bucket, prefix = url[5:].split("/", 1)
        store = obstore.store.GCSStore(bucket=bucket, prefix=prefix, skip_signature=True)
        return xr.open_zarr(zarr.storage.ObjectStore(store), chunks=None)
    return xr.open_zarr(url, chunks=None)


def crop(ds, box=REGION_PRESETS["france"]):
    """Crop lazily across Greenwich before sorting into ascending signed coordinates."""
    ds = ds.rename({k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in ds.dims})
    iy, ix = crop_indices(box, ds.lat.values, ds.lon.values)
    ds = ds.isel(lat=iy, lon=ix)
    return ds.assign_coords(lon=to_signed_lon(ds.lon.values)).sortby("lon").sortby("lat")


def read_cost(rate, times, box):
    """Estimate touched chunks from metadata, not compression-dependent wire bytes."""
    lat_name = "latitude" if "latitude" in rate.dims else "lat"
    lon_name = "longitude" if "longitude" in rate.dims else "lon"
    iy, ix = crop_indices(box, rate[lat_name].values, rate[lon_name].values)
    stamps = pd.DatetimeIndex(np.concatenate([
        pd.date_range(pd.Timestamp(t) - pd.Timedelta(minutes=45), t, freq="15min").values for t in times
    ])).unique()
    it = rate.get_index("time").get_indexer(stamps)
    if not len(it) or np.any(it < 0):
        raise ValueError("requested hourly endpoints fall outside the PARDIG archive")
    indexes = {"time": it, lat_name: iy, lon_name: ix}
    chunks = rate.encoding["chunks"]
    counts, lengths = [], []
    for dim, chunk in zip(rate.dims, chunks):
        touched = np.unique(indexes[dim] // chunk)
        counts.append(len(touched))
        lengths.append(sum(min(chunk, rate.sizes[dim] - int(i) * chunk) for i in touched))
    return {"chunk_shape": list(chunks), "chunks_touched": int(np.prod(counts)),
            "decoded_bytes_upper_bound": int(np.prod(lengths)) * rate.dtype.itemsize,
            "note": "compressed transfer bytes depend on data; coordinate/metadata reads excluded"}


def plan_pardig(times, box=REGION_PRESETS["france"], layout="auto"):
    """Compare the two layouts from the developer guide before reading rainfall."""
    candidates = {"spatial": PARDIG_NRT, "timeseries": PARDIG_TIMESERIES}
    if layout not in ("auto", *candidates):
        raise ValueError("layout must be auto, spatial or timeseries")
    costs = {}
    for name, url in candidates.items():
        with open_store(url) as ds:
            costs[name] = {"url": url, **read_cost(ds.precipitation, times, box)}
    chosen = min(costs, key=lambda name: costs[name]["decoded_bytes_upper_bound"]) if layout == "auto" else layout
    return {"chosen": chosen, "candidates": costs}


def hourly_pardig(rate: xr.DataArray, times) -> xr.DataArray:
    """Paper A.3.8: 1/4 * sum(rate[t-45m:t]) over (t-1h,t], in metres.

    Missing timestamps raise; missing pixels stay missing, never become dry hours.
    """
    if rate.attrs.get("units") not in ("m h**-1", "m/h"):
        raise ValueError("PARDIG precipitation must be instantaneous rate in m/h")
    times = pd.DatetimeIndex(times)
    if times.tz is not None or not times.equals(times.floor("h")):
        raise ValueError("times must be whole UTC hours without a timezone")
    if len(times) == 0:
        raise ValueError("at least one hour is required")
    # Load each touched compressed archive chunk once for this request, instead
    # of re-reading the boundary chunks for every successive hourly sum.
    needed = pd.DatetimeIndex(np.concatenate([
        pd.date_range(t - pd.Timedelta(minutes=45), t, freq="15min").values for t in times
    ])).unique().sort_values()
    rate = rate.sel(time=needed).compute(scheduler="synchronous")
    frames = []
    for t in times:
        stamps = pd.date_range(t - pd.Timedelta(minutes=45), t, freq="15min")
        samples = rate.sel(time=stamps)
        frames.append((samples.sum("time", skipna=False) * 0.25).expand_dims(time=[t]))
    result = xr.concat(frames, dim="time").astype("float32")
    result.attrs = {"units": "m", "interval": "(time-1h,time]", "source": PARDIG_NRT}
    return result


class PARDIG:
    """Earth2Studio DataSource protocol: (time, variable) -> DataArray.

    Supports ``tp`` (one-hour depth, m). This regional adapter uses signed longitude.
    The default historical store covers 2016–2025; pass a prepared local store too.
    """
    def __init__(self, url=PARDIG_NRT, box=REGION_PRESETS["france"], dataset=None):
        self.ds = crop(open_store(url) if dataset is None else dataset, box)

    def __call__(self, time, variable):
        variables = [variable] if isinstance(variable, str) else list(variable)
        if variables != ["tp"]:
            raise ValueError("PARDIG supports only ['tp'] (hourly depth in m)")
        times = [time] if isinstance(time, (str, np.datetime64, pd.Timestamp)) else np.atleast_1d(time)
        return hourly_pardig(self.ds.precipitation, times).expand_dims(variable=["tp"]).transpose(
            "time", "variable", "lat", "lon"
        )


def era5_fields(times, variables, backend="arco", box=REGION_PRESETS["france"]):
    """Read hourly ERA5 with direct xarray access or the optional Earth2Studio helper."""
    # Include a halo for interpolation onto PARDIG's boundary cells.
    halo = (box[0] - 1, box[1] + 1, box[2] - 1, box[3] + 1)
    if backend == "earth2studio":
        from earth2studio.data import ARCO_ERA5
        return crop(ARCO_ERA5()(list(pd.DatetimeIndex(times).to_pydatetime()), variables), halo)
    if backend != "arco":
        raise ValueError("backend must be arco or earth2studio")
    ds = crop(open_store(ERA5_ARCO)[[ERA5_NAMES[v] for v in variables]].sel(time=times), halo)
    arrays = [ds[ERA5_NAMES[v]].sel(time=times).compute(scheduler="synchronous").expand_dims(variable=[v]) for v in variables]
    return xr.concat(arrays, dim="variable").transpose("time", "variable", "lat", "lon")


def build_pairs(times, variables=("tp", "t2m", "d2m", "u10m", "v10m", "msl"),
                backend="arco", pardig_url=PARDIG_NRT, box=REGION_PRESETS["france"], target=None,
                forcing_hours=1):
    """Pair ERA5 0.25° hourly predictors with PARDIG 0.1° hourly rainfall."""
    if not variables or variables[0] != "tp" or len(set(variables)) != len(variables):
        raise ValueError("predictors must be unique and start with tp")
    truth = (PARDIG(pardig_url, box)(times, ["tp"]).sel(variable="tp", drop=True)
             if target is None else target.sel(time=times))
    if forcing_hours > 1:
        if len(times) % forcing_hours:
            raise ValueError("predictor request must contain complete forcing windows")
        fields = era5_fields(times, ["tp"], backend, box)
        if len(variables) > 1:
            endpoints = pd.DatetimeIndex(times)[forcing_hours - 1::forcing_hours]
            context = era5_fields(endpoints, list(variables[1:]), backend, box)
            # Endpoint states are all the model consumes: avoid downloading five
            # intermediate global fields for every six-hour window.
            index = np.repeat(np.arange(len(endpoints)), forcing_hours)
            context = context.isel(time=index).assign_coords(time=pd.DatetimeIndex(times))
            fields = xr.concat([fields, context], dim="variable")
    else:
        fields = era5_fields(times, list(variables), backend, box)
    fields = fields.interp(lat=truth.lat, lon=truth.lon).astype("float32")
    # Both ERA5 tp and aggregated PARDIG are depths in m over the preceding hour.
    fields.loc[dict(variable="tp")] = fields.sel(variable="tp") * 1000.0
    fields.loc[dict(variable="tp")] = fields.sel(variable="tp").clip(min=0)
    pack = xr.Dataset({"predictors": fields, "target": truth * 1000.0})
    pack.target.attrs = {"units": "mm", "interval": "(time-1h,time]"}
    pack.attrs = {
        "target_source": pardig_url, "coarse_source": f"ERA5 via {backend}",
        "task": "hourly precipitation downscaling", "target_resolution_degrees": 0.1,
        "precipitation_units": "mm", "interval": "(time-1h,time]",
        "predictor_units": "tp:mm,t2m:K,d2m:K,u10m:m/s,v10m:m/s,msl:Pa",
        "satellite_conditioning": "none; future valid-time imagery would leak observations",
    }
    return pack


def interval_features(depth: xr.DataArray, hours: int) -> xr.DataArray:
    """Condition hourly output on an h-hour forecast accumulation and within-window phase.

    Input must be consecutive hourly maps. Each full h-hour window shares its total
    and changes phase. This simulates the information available from h-hour forecasts;
    it does not linearly interpolate rain. Incomplete windows are rejected.
    """
    times = pd.DatetimeIndex(depth.time.values)
    if hours < 1 or len(times) % hours or len(times) == 0:
        raise ValueError("hour count must be positive and divide the sample count")
    if not np.all(np.diff(times.values) == np.timedelta64(1, "h")):
        raise ValueError("input hours must be consecutive")
    if (times[0].hour - 1) % hours:
        raise ValueError("first hour must end one hour after a forecast window boundary")
    values = depth.values.reshape(-1, hours, *depth.shape[1:])
    total = np.repeat(values.sum(1, keepdims=True), hours, axis=1).reshape(depth.shape)
    phase = np.broadcast_to(np.tile(np.arange(1, hours + 1) / hours, len(times) // hours)[:, None, None], depth.shape)
    return xr.concat([
        xr.DataArray(total / hours, coords=depth.coords, dims=depth.dims),
        xr.DataArray(phase, coords=depth.coords, dims=depth.dims),
    ], xr.IndexVariable("variable", np.array(["tp", "phase"], dtype=object))).transpose("time", "variable", "lat", "lon")


def condition_on_intervals(pack: xr.Dataset, hours: int) -> xr.Dataset:
    """Hide intermediate ERA5 states to match h-hour global-forecast forcing."""
    rain = interval_features(pack.predictors.sel(variable="tp", drop=True), hours)
    context = []
    for variable in pack.variable.values[1:]:
        a = pack.predictors.sel(variable=variable, drop=True)
        # Only the endpoint state is used, as provided by the global forecaster.
        v = a.values.reshape(-1, hours, *a.shape[1:])
        repeated = np.repeat(v[:, -1:, ...], hours, axis=1).reshape(a.shape)
        context.append(xr.DataArray(repeated, coords=a.coords, dims=a.dims).expand_dims(variable=[variable]))
    out = pack.drop_vars("predictors").drop_dims("variable")
    out["predictors"] = xr.concat([rain.sel(variable=["tp"]), *context, rain.sel(variable=["phase"])], dim="variable").astype("float32")
    out.attrs = {**pack.attrs, "forecast_interval_hours": hours,
                 "forcing": "interval-mean rainfall and endpoint atmospheric state"}
    return out


def forecast_pack(fields: xr.DataArray, lat, lon, hours=6, accumulation="cumulative"):
    """Convert one global forecast/member to the training schema.

    ``fields``: (lead_time, variable, lat, lon), scalar init ``time`` coordinate;
    ``tp`` in metres, either cumulative since initialization (include lead 0), or
    per-interval depths (first lead = hours). Other fields are endpoint states.
    Call separately per initialization/member; never difference across forecast runs.
    """
    if accumulation not in ("cumulative", "interval"):
        raise ValueError("accumulation must explicitly be cumulative or interval")
    if fields.dims != ("lead_time", "variable", "lat", "lon") or fields.time.ndim != 0:
        raise ValueError("select one initialization/member; expected lead_time, variable, lat, lon and scalar time")
    if "tp" not in fields["variable"].values:
        raise ValueError("forecast requires tp in metres")
    leads = fields.lead_time.values / np.timedelta64(1, "h")
    if hours < 1 or not float(hours).is_integer() or not np.all(np.diff(leads) == hours):
        raise ValueError("forecast lead spacing must match the trained interval")
    start = 0 if accumulation == "cumulative" else hours
    if not len(leads) or leads[0] != start:
        raise ValueError(f"first lead must be {start} hours for {accumulation} rainfall")
    if accumulation == "cumulative" and len(leads) < 2:
        raise ValueError("need at least two cumulative endpoints")
    fields = fields.interp(lat=lat, lon=lon)
    rain = fields.sel(variable="tp", drop=True).values * 1000
    if accumulation == "cumulative":
        rain = np.diff(rain, axis=0)
        fields = fields.isel(lead_time=slice(1, None))
        leads = leads[1:]
    if not np.isfinite(rain).all() or np.any(rain < -1e-3):
        raise ValueError("missing rainfall or accumulation reset; check forecast step semantics")
    rain = np.maximum(rain, 0)
    times = np.concatenate([fields.time.values + np.arange(int(end) - hours + 1, int(end) + 1).astype("timedelta64[h]") for end in leads])
    variables = ["tp", *[str(v) for v in fields["variable"].values if v != "tp"], "phase"]
    arrays = [np.repeat(rain / hours, hours, axis=0)]
    arrays += [np.repeat(fields.sel(variable=v).values, hours, axis=0) for v in variables[1:-1]]
    phase = np.tile(np.arange(1, hours + 1) / hours, len(leads))[:, None, None]
    arrays += [np.broadcast_to(phase, arrays[0].shape)]
    return xr.Dataset({"predictors": (("time", "variable", "lat", "lon"), np.stack(arrays, axis=1).astype("float32"))},
                      coords={"time": times, "variable": variables, "lat": lat, "lon": lon},
                      attrs={"precipitation_units": "mm", "interval": "(time-1h,time]",
                             "forecast_interval_hours": hours, "forcing": "global forecast"})
