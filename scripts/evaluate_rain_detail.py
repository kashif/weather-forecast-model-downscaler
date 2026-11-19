"""Compare paired rainfall fine-tunes on PARDIG and fresh French gauge dates."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, uniform_filter
import torch
import xarray as xr
import yaml

from downscale.precip_data import ERA5_NAMES, crop, open_store
from downscale.precipitation import detail_band_crps, score
from downscale.rain_alerts import contingency, precision_recall
from downscale.rain_resume import atomic_json
from downscale.sources import ERA5_ARCO
from downscale.station_interp import GridInterpolator
from scripts.check_rain_gauges import aligned_observations, quality_control
from scripts.precipitation import write_pack
from scripts.precipitation_live import conditional_ensemble
from scripts.precipitation_year import day_pack, hour_ends


OUT = Path("runs/rain_detail")
GAUGES = Path("data/precipitation/gauge_detail")
CHECKPOINTS = {
    "annual": Path("runs/precipitation_year/model.pt"),
    "control": OUT / "control/model.pt",
    "detail": OUT / "detail/model.pt",
}
THRESHOLDS = (5., 10.)
MEMBERS = 32
BUDGET = 0.001


def gauge_dates():
    config = yaml.safe_load(Path("config_rain_detail_gauges.yaml").read_text())
    return config, [pd.Timestamp(day) for day in config["splits"]["test"]]


def hours(day):
    return hour_ends(day, day + pd.Timedelta(days=1))


def prepare_forcing():
    config = yaml.safe_load(Path("config_precipitation_year.yaml").read_text())
    cp = torch.load(CHECKPOINTS["annual"], map_location="cpu", weights_only=True)
    grid = xr.Dataset(coords={key: cp[key].numpy() for key in ("lat", "lon")})
    south, north, west, east = config["region"]
    source = crop(open_store(ERA5_ARCO)[list(ERA5_NAMES.values())],
                  (south - 1, north + 1, west - 1, east + 1))
    for day in gauge_dates()[1]:
        path = OUT / "forcing" / f"{day:%Y%m%d}.zarr"
        if not path.exists():
            write_pack(day_pack(source, hours(day), None, grid, config), path)
        print(f"Gauge forcing ready for {day:%Y-%m-%d}", flush=True)


def grid_day(rain, truth):
    result = score(rain, truth)
    # Batch the member-pair calculation to limit peak memory at 32 members.
    result["detail_crps_mm"] = float(np.mean([
        detail_band_crps(torch.from_numpy(rain[i:i + 4]), torch.from_numpy(truth[i:i + 4])).item()
        for i in range(0, len(truth), 4)
    ]))
    for threshold in THRESHOLDS:
        probability = (rain > threshold).mean(axis=1)
        event = (truth > threshold).astype(float)
        fractions = {}
        for size in (1, 5, 9):
            pred = uniform_filter(probability, size=(1, size, size))
            obs = uniform_filter(event, size=(1, size, size))
            denominator = float(np.sum(pred**2 + obs**2))
            fractions[str(size)] = dict(squared_error=float(np.sum((pred - obs)**2)),
                                        denominator=denominator)
        result[f"fss_components_{threshold:g}"] = fractions
    return result


def gauge_day(rain, interpolator, thresholds):
    """Same gauge and observed event throughout, with wider forecast search windows."""
    probabilities = {}
    for threshold in thresholds:
        event = rain > threshold
        spatial = maximum_filter(event, size=(1, 1, 5, 5), mode="constant")
        for name, fields in (
            ("exact", event),
            ("within_2_grid_cells", spatial),
            ("within_2_grid_cells_and_1_hour", maximum_filter(spatial, size=(3, 1, 1, 1), mode="constant")),
        ):
            probability = xr.DataArray(fields.mean(axis=1), dims=("time", "lat", "lon"))
            probabilities[threshold, name] = interpolator.apply(probability).values[1:-1]
    return probabilities


def gauge_score(probability, observed):
    curve = precision_recall(probability, observed, BUDGET)
    return dict(average_precision=curve["average_precision"],
                brier=float(np.mean((probability - observed)**2)),
                at_one_member=contingency(probability, observed, 1 / MEMBERS),
                curve_at_matched_fpr=curve["recall_at_fpr_curve_diagnostic"])


def evaluate():
    torch.set_num_threads(4)
    config, dates = gauge_dates()
    forcing = xr.open_zarr(OUT / "forcing" / f"{dates[0]:%Y%m%d}.zarr")
    obs_path = GAUGES / "observations.parquet"
    times = pd.DatetimeIndex(np.concatenate([hours(day).values for day in dates]))
    rows, stations, qc = quality_control(
        pd.read_parquet(obs_path), times,
        [float(forcing.lat.min()), float(forcing.lat.max()), float(forcing.lon.min()), float(forcing.lon.max())],
        config["gauge_min_completeness"])
    interpolator = GridInterpolator.build(forcing.lat, forcing.lon, stations.lat, stations.lon, "nearest")
    obs = np.concatenate([aligned_observations(rows, stations, hours(day))[1:-1] for day in dates])
    valid = np.isfinite(obs)
    cases = sorted(Path("data/precipitation/gpd_calibration/test").glob("*/pairs.zarr"))
    if len(cases) != 4:
        raise ValueError("Expected the four cached seasonal 2024 cases")
    report = dict(members=MEMBERS, gauge_dates=[str(day.date()) for day in dates],
                  gauge_quality=qc, station_hours=int(valid.sum()),
                  observations_sha256=hashlib.sha256(obs_path.read_bytes()).hexdigest(),
                  models={})
    for name, ckpt in CHECKPOINTS.items():
        results = dict(sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(), pardig={}, gauges={})
        for path in cases:
            pack = xr.open_zarr(path).load()
            rain = conditional_ensemble(ckpt, pack, MEMBERS).precipitation.values
            results["pardig"][path.parent.name] = grid_day(rain, pack.target.values)
        daily = []
        for day in dates:
            pack = xr.open_zarr(OUT / "forcing" / f"{day:%Y%m%d}.zarr").load()
            rain = conditional_ensemble(ckpt, pack, MEMBERS).precipitation.values
            daily.append(gauge_day(rain, interpolator, THRESHOLDS))
            print(f"{name} scored {day:%Y-%m-%d}", flush=True)
        for threshold, window in daily[0]:
            probability = np.concatenate([day[threshold, window] for day in daily])[valid]
            scores = gauge_score(probability, obs[valid] > threshold)
            scores["per_date"] = {}
            for index, day in enumerate(dates):
                day_slice = slice(22 * index, 22 * (index + 1))
                keep = valid[day_slice]
                scores["per_date"][str(day.date())] = contingency(
                    daily[index][threshold, window][keep], obs[day_slice][keep] > threshold, 1 / MEMBERS)
            results["gauges"].setdefault(str(threshold), {})[window] = scores
        results["pardig_summary"] = {
            metric: float(np.mean([day[metric] for day in results["pardig"].values()]))
            for metric in ("crps_mm", "detail_crps_mm", "rmse_mm")
        }
        for threshold in THRESHOLDS:
            key = f"fss_components_{threshold:g}"
            aggregate = {}
            for size in ("1", "5", "9"):
                num = sum(day[key][size]["squared_error"] for day in results["pardig"].values())
                den = sum(day[key][size]["denominator"] for day in results["pardig"].values())
                aggregate[size] = 1 - num / den if den else None
            results["pardig_summary"][f"fss_{threshold:g}"] = aggregate
        report["models"][name] = results
        if name == "annual":
            atomic_json(OUT / "baseline.json", report)
    atomic_json(OUT / "results.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare_forcing() if args.prepare else evaluate()
