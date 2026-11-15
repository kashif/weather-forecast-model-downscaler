"""Regional probabilistic rainfall downscaler with physical-unit multiscale CRPS."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch import nn

from .downscaler import UNet, multiscale_fair_crps, time_features


@dataclass
class RainConfig:
    base: int = 16
    depth: int = 3
    noise_factor: int = 8
    members: int = 4
    window_hours: int = 6
    temporal_context: bool = False
    architecture: str = "spatial"
    sequence_hours: int = 12
    terrain: bool = False


def check_pack(ds: xr.Dataset, target=True):
    if ds.predictors.dims != ("time", "variable", "lat", "lon"):
        raise ValueError("predictors must have dimensions time, variable, lat, lon")
    if str(ds.variable.values[0]) != "tp":
        raise ValueError("the first predictor must be tp, hourly or interval-mean depth in mm")
    if ds.attrs.get("precipitation_units") != "mm":
        raise ValueError("pack must declare precipitation_units: mm")
    if target and ds.target.dims != ("time", "lat", "lon"):
        raise ValueError("target must have dimensions time, lat, lon")
    times = ds.time.values
    if len(times) == 0 or len(np.unique(times)) != len(times) or np.any(np.diff(times) <= np.timedelta64(0, "s")):
        raise ValueError("times must be nonempty, unique and increasing")
    for coord in ("lat", "lon"):
        if ds.sizes[coord] < 2 or not np.all(np.diff(ds[coord].values) > 0):
            raise ValueError(f"{coord} must be strictly increasing with at least two cells")
    if not np.isfinite(ds.predictors.values).all():
        raise ValueError("predictors contain missing/nonfinite values")
    if np.any(ds.predictors.sel(variable="tp").values < 0):
        raise ValueError("rainfall predictors must be nonnegative")
    if target and (not np.isfinite(ds.target.values).all() or np.any(ds.target.values < 0)):
        raise ValueError("rainfall targets must be finite and nonnegative")


def features(ds: xr.Dataset, mean=None, std=None, temporal_context=False, static=None):
    """Normalise using training-only statistics, append geography and calendar."""
    if temporal_context:
        from .rain_temporal import context_inputs
        data, names = context_inputs(ds)
        for i, name in enumerate(names):
            if name == "tp" or name.endswith("_tp"):
                data[:, i] = np.log1p(data[:, i])
    else:
        data = ds.predictors.values.astype("float32")
        data[:, 0] = np.log1p(data[:, 0])
    if mean is None:
        mean = data.mean((0, 2, 3), keepdims=True)
        std = np.maximum(data.std((0, 2, 3), keepdims=True), 1e-4)
    elif data.shape[1] != mean.shape[1]:
        raise ValueError("checkpoint feature layout differs; use matching code or retrain")
    data -= mean
    data /= std
    n, _, h, w = data.shape
    lat, lon = np.meshgrid(ds.lat.values, ds.lon.values, indexing="ij")
    geo = np.broadcast_to(np.stack([lat / 90, lon / 180]).astype("float32")[None], (n, 2, h, w))
    calendar = np.broadcast_to(time_features(ds.time.values).astype("float32")[:, :, None, None], (n, 4, h, w))
    arrays = [data, geo, calendar]
    if static is not None:
        from .rain_terrain import CHANNELS
        static = np.asarray(static, dtype="float32")
        if static.shape != (len(CHANNELS), h, w) or not np.isfinite(static).all():
            raise ValueError("static terrain must have the expected channels on the rainfall grid")
        arrays.append(np.broadcast_to(static[None], (n, *static.shape)))
    return np.concatenate(arrays, 1).astype("float32", copy=False), mean, std


class RainNet(nn.Module):
    def __init__(self, channels: int, cfg: RainConfig):
        super().__init__()
        if cfg.noise_factor < 1 or cfg.members < 2 or cfg.depth < 1 or cfg.window_hours < 1:
            raise ValueError("positive depth/noise_factor and at least two members required")
        self.cfg = cfg
        if cfg.architecture not in ("spatial", "joint"):
            raise ValueError("architecture must be spatial or joint")
        if cfg.architecture == "joint":
            if cfg.sequence_hours < 2 * cfg.window_hours or cfg.sequence_hours % (2 * cfg.window_hours):
                raise ValueError("joint sequence_hours must be a multiple of twice window_hours")
            from .rain_sequence import TrajectoryUNet
            self.unet = TrajectoryUNet(channels + 1, cfg.base, cfg.depth, cfg.sequence_hours)
        else:
            self.unet = UNet(channels + 1, 1, cfg.base, cfg.depth, "bilinear")

    def forward(self, x, baseline, members=None, noise=None):
        """Return nonnegative hourly depths (batch, members, lat, lon), in mm."""
        m = self.cfg.members if members is None else members
        if m < 1:
            raise ValueError("members must be positive")
        b, _, h, w = x.shape
        if b % self.cfg.window_hours:
            raise ValueError("batch must contain complete forecast windows")
        if noise is None and (self.cfg.temporal_context or self.cfg.architecture == "joint"):
            from .rain_temporal import trajectory_noise
            noise = trajectory_noise(np.arange(b).astype("datetime64[h]"), m, h, w, self.cfg.noise_factor, self.cfg.window_hours)
        if noise is None:
            noise = torch.randn(b // self.cfg.window_hours, m, 1, max(1, h // self.cfg.noise_factor),
                                max(1, w // self.cfg.noise_factor), device=x.device)
            noise = noise[:, None].expand(-1, self.cfg.window_hours, -1, -1, -1, -1)
        elif noise.shape[:3] != (b, m, 1):
            raise ValueError("noise must have shape time, members, 1, noise_lat, noise_lon")
        noise = noise.to(x.device).reshape(b * m, 1, *noise.shape[-2:])
        noise = F.interpolate(noise, (h, w), mode="bilinear", align_corners=False)
        inputs = torch.cat([x.repeat_interleave(m, 0), noise], 1)
        multiple = 2 ** self.cfg.depth
        inputs = F.pad(inputs, (0, (-w) % multiple, 0, (-h) % multiple), mode="replicate")
        decoded = self.unet(inputs, members=m) if self.cfg.architecture == "joint" else self.unet(inputs)
        correction = decoded[:, 0, :h, :w].reshape(b, m, h, w)
        # A small offset gives initially dry pixels a gradient. Upper bound prevents overflow.
        log_depth = (torch.log1p(baseline[:, None]) + 0.01 + correction).clamp(0, np.log1p(500.0))
        return torch.expm1(log_depth)


def rain_loss(pred, truth, window_hours=1, sequence_hours=None, temporal_weight=0):
    factors = [f for f in (2, 4, 8) if f <= min(truth.shape[-2:])]
    from .downscaler import fair_crps
    # WN3 A.1.2 explicitly adds domain-mean CRPS at weight 0.3.
    loss = multiscale_fair_crps(pred, truth, factors=factors)
    loss = loss + 0.3 * fair_crps(pred.mean((-2, -1), keepdim=True), truth.mean((-2, -1), keepdim=True))
    if window_hours > 1:
        if len(truth) % window_hours:
            raise ValueError("loss requires complete forecast windows")
        p = pred.reshape(-1, window_hours, *pred.shape[1:]).sum(1)
        t = truth.reshape(-1, window_hours, *truth.shape[1:]).sum(1)
        loss = loss + 0.3 * fair_crps(p, t) / window_hours
    if temporal_weight:
        length = sequence_hours or len(truth)
        if length < 2 or len(truth) % length:
            raise ValueError("temporal loss needs complete sequences of at least two hours")
        p = pred.reshape(-1, length, *pred.shape[1:])
        t = truth.reshape(-1, length, *truth.shape[1:])
        dp, dt = p[:, 1:] - p[:, :-1], t[:, 1:] - t[:, :-1]
        loss = loss + temporal_weight * multiscale_fair_crps(dp.reshape(-1, *pred.shape[1:]), dt.reshape(-1, *truth.shape[1:]), factors=factors)
    return loss


def score(pred, truth):
    """Physical-unit scores against the supplied target; no independent-skill claim."""
    from .evaluation import fair_crps
    pred, truth = np.asarray(pred), np.asarray(truth)
    mean = pred.mean(1)
    result = {"rmse_mm": float(np.sqrt(np.mean((mean - truth) ** 2))),
              "bias_mm": float(np.mean(mean - truth)),
              "crps_mm": float(fair_crps(pred, truth, axis=1).mean()) if pred.shape[1] > 1 else float(np.abs(mean - truth).mean())}
    # WN3 reports actual finite-ensemble CRPS at evaluation, whereas fair CRPS
    # estimates the underlying generator score. Report both, with explicit names.
    skill = float(np.abs(pred - truth[:, None]).mean())
    result["ensemble_crps_mm"] = result["crps_mm"] + (skill - result["crps_mm"]) / pred.shape[1]
    for threshold in (0.1, 1.0, 5.0, 10.0):
        observed = truth >= threshold
        prob = (pred >= threshold).mean(1)
        event = prob >= 0.5
        union = np.logical_or(event, observed).sum()
        result[f"brier_{threshold:g}mm"] = float(np.mean((prob - observed) ** 2))
        result[f"csi_{threshold:g}mm"] = float(np.logical_and(event, observed).sum() / union) if union else None
        result[f"observed_events_{threshold:g}mm"] = int(observed.sum())
        # Reliability of event probabilities, including empty-bin counts.
        reliability = []
        for lo, hi in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
            mask = (prob >= lo) & ((prob < hi) if hi < 1 else (prob <= hi))
            reliability.append({"n": int(mask.sum()), "mean_probability": float(prob[mask].mean()) if mask.any() else None,
                                "observed_frequency": float(observed[mask].mean()) if mask.any() else None})
        result[f"reliability_{threshold:g}mm"] = reliability
        # Neighbourhood FSS for the ensemble-median event footprint. Padding is
        # excluded, so edge neighbourhoods only use points within the domain.
        for size in (1, 3, 9):
            if size > min(truth.shape[-2:]):
                continue
            a = F.avg_pool2d(torch.tensor(event[:, None], dtype=torch.float32), size, stride=1, padding=size // 2, count_include_pad=False).numpy()
            b = F.avg_pool2d(torch.tensor(observed[:, None], dtype=torch.float32), size, stride=1, padding=size // 2, count_include_pad=False).numpy()
            denom = np.mean(a * a + b * b)
            result[f"fss_{threshold:g}mm_{size}px"] = float(1 - np.mean((a - b) ** 2) / denom) if denom else None
    return result


def evaluate_rain(prediction: xr.Dataset, pairs: xr.Dataset, interior_margin=8):
    """Score exactly aligned hourly predictions and accumulated window totals."""
    check_pack(pairs)
    rain = prediction.precipitation
    if rain.dims != ("time", "sample", "lat", "lon") or rain.attrs.get("units") != "mm":
        raise ValueError("prediction must be time, sample, lat, lon in mm")
    for name in ("time", "lat", "lon"):
        if not np.array_equal(prediction[name], pairs[name]):
            raise ValueError(f"prediction and target {name} differ")
    pred, truth = rain.values, pairs.target.values
    if not np.isfinite(pred).all() or np.any(pred < 0):
        raise ValueError("predicted rain must be finite and nonnegative")
    baseline = pairs.predictors.sel(variable="tp").values[:, None]
    window = int(pairs.attrs["forecast_interval_hours"])
    if len(truth) % window:
        raise ValueError("evaluation requires complete forcing windows")
    report = {
        "reference": "PARDIG proxy; not independent radar/gauge verification",
        "start": str(pairs.time.values[0]), "end": str(pairs.time.values[-1]),
        "hours": len(truth), "members": pred.shape[1], "domain": "rectangular France context; uniform pixel weights",
        "hourly": {"model": score(pred, truth), "baseline": score(baseline, truth)},
        "accumulations": {},
    }
    from .rain_temporal import hourly_baseline
    smooth = hourly_baseline(pairs)
    report["hourly"]["smooth_coarse"] = score(smooth[:, None], truth)
    report["temporal"] = temporal_scores(pred, truth, baseline[:, 0], smooth, pairs.time.values, window)
    # A context ring is useful for regional models. Report the interior
    # separately so edge-padding behaviour cannot hide behind a domain mean.
    if interior_margin > 0 and min(truth.shape[-2:]) > 2 * interior_margin:
        core = (slice(None), slice(interior_margin, -interior_margin), slice(interior_margin, -interior_margin))
        pc = pred[:, :, interior_margin:-interior_margin, interior_margin:-interior_margin]
        bc = baseline[:, :, interior_margin:-interior_margin, interior_margin:-interior_margin]
        report["interior_hourly"] = {"excluded_context_cells": interior_margin,
                                      "model": score(pc, truth[core]), "baseline": score(bc, truth[core])}
    for hours in sorted({window, 24}):
        if len(truth) % hours or not np.all(np.diff(pairs.time.values.reshape(-1, hours), axis=1) == np.timedelta64(1, "h")):
            continue
        p = pred.reshape(-1, hours, *pred.shape[1:]).sum(1)
        t = truth.reshape(-1, hours, *truth.shape[1:]).sum(1)
        b = baseline.reshape(-1, hours, *baseline.shape[1:]).sum(1)
        report["accumulations"][str(hours)] = {"units": f"mm/{hours}h", "model": score(p, t), "baseline": score(b, t)}
    # Paired per-window scores make timing/volume failures inspectable. A few
    # windows from one event do not justify independent bootstrap confidence claims.
    report["per_window"] = []
    for i in range(0, len(truth), window):
        end = i + window
        report["per_window"].append({"start": str(pairs.time.values[i]),
                                      "model_crps_mm": score(pred[i:end], truth[i:end])["crps_mm"],
                                      "baseline_crps_mm": score(baseline[i:end], truth[i:end])["crps_mm"]})
    return report


def temporal_scores(pred, truth, coarse, smooth, times, hours=6):
    """Compare boundary changes with ordinary hourly changes and target increments."""
    from .evaluation import fair_crps
    import pandas as pd
    times = pd.DatetimeIndex(times)
    consecutive = np.diff(times.values) == np.timedelta64(1, "h")
    boundary = (np.arange(1, len(times)) % hours == 0) & consecutive
    ordinary = ~boundary & consecutive
    target_changes = np.diff(truth, axis=0)
    result = {"boundary_count": int(boundary.sum()), "within_window_count": int(ordinary.sum()),
              "note": "physical changes can be rapid; compare against the target, not zero change"}
    for name, values in {"model_mean": pred.mean(1), "model_members": pred,
                         "target": truth, "coarse": coarse, "smooth_coarse": smooth}.items():
        delta = np.diff(values, axis=0)
        magnitude = np.abs(delta).reshape(len(delta), -1).mean(1)
        b = float(magnitude[boundary].mean()) if boundary.any() else None
        w = float(magnitude[ordinary].mean()) if ordinary.any() else None
        result[name] = {"boundary_abs_change_mm": b, "within_window_abs_change_mm": w,
                        "boundary_to_within_ratio": b / w if b is not None and w else None}
        if consecutive.all() and len(values) >= 2 * hours:
            centered = values - values.mean(axis=0, keepdims=True)
            power = (np.abs(np.fft.rfft(centered, axis=0)) ** 2).reshape(len(values) // 2 + 1, -1).mean(1)
            frequency = int(round(len(values) / hours))
            total = float(power[1:].sum())
            result[name]["six_hour_power_fraction"] = float(power[frequency] / total) if total else None
    delta = np.diff(pred, axis=0)
    crps = fair_crps(delta, target_changes, axis=1).reshape(len(delta), -1).mean(1)
    result["increment_crps_mm"] = float(crps[consecutive].mean()) if consecutive.any() else None
    result["boundary_increment_crps_mm"] = float(crps[boundary].mean()) if boundary.any() else None
    result["mean_increment_mae_mm"] = float(np.abs(delta.mean(1) - target_changes)[consecutive].mean()) if consecutive.any() else None
    return result


def save_model(path, net, ds, mean, std, static=None, static_metadata=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"format": "france-rain-v1", "config": asdict(net.cfg),
                "variables": ds.variable.values.tolist(), "mean": torch.tensor(mean), "std": torch.tensor(std),
                "lat": torch.tensor(ds.lat.values), "lon": torch.tensor(ds.lon.values),
                "data_attrs": dict(ds.attrs), "state_dict": net.state_dict()}
    if net.cfg.terrain:
        from .rain_terrain import CHANNELS
        if static is None:
            raise ValueError("terrain-enabled models require embedded static covariates")
        checkpoint.update(static=torch.tensor(static, dtype=torch.float32), static_channels=CHANNELS,
                          static_metadata=static_metadata or {})
    torch.save(checkpoint, path)


def predict(path, ds, device="cpu", members=8, batch=4, noise_seed=None):
    check_pack(ds, target=False)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format"] != "france-rain-v1" or ds.variable.values.tolist() != checkpoint["variables"]:
        raise ValueError("checkpoint format or predictor channels differ")
    for name in ("lat", "lon"):
        if not np.array_equal(ds[name].values, checkpoint[name].numpy()):
            raise ValueError(f"{name} differs from the training grid; regrid predictors first")
    for name in ("forecast_interval_hours", "interval"):
        if ds.attrs.get(name) != checkpoint["data_attrs"].get(name):
            raise ValueError(f"{name} differs from the training convention")
    cfg = RainConfig(**checkpoint["config"])
    static = None
    if cfg.terrain:
        from .rain_terrain import CHANNELS
        if checkpoint.get("static_channels") != CHANNELS or "static" not in checkpoint:
            raise ValueError("terrain checkpoint is missing static covariates")
        static = checkpoint["static"].numpy()
    x, _, _ = features(ds, checkpoint["mean"].numpy(), checkpoint["std"].numpy(), cfg.temporal_context, static)
    net = RainNet(x.shape[1], cfg).to(device).eval()
    net.load_state_dict(checkpoint["state_dict"])
    window = net.cfg.window_hours
    if len(x) % window:
        raise ValueError("inference requires complete forecast windows")
    batch = max(window, (batch // window) * window)
    noise = None
    baseline = ds.predictors.values[:, 0]
    if cfg.temporal_context or cfg.architecture == "joint":
        from .rain_temporal import hourly_baseline, trajectory_noise
        baseline = hourly_baseline(ds)
        generator = None if noise_seed is None else torch.Generator().manual_seed(noise_seed)
        noise = trajectory_noise(ds.time.values, members, len(ds.lat), len(ds.lon), cfg.noise_factor, window, generator=generator)
    outputs = []
    with torch.no_grad():
        if cfg.architecture == "joint":
            from .rain_sequence import predict_overlapping
            outputs.append(predict_overlapping(net, x, baseline, noise, ds.time.values, device, members))
        else:
            for i in range(0, len(x), batch):
                outputs.append(net(torch.tensor(x[i:i + batch], device=device),
                                   torch.tensor(baseline[i:i + batch], dtype=torch.float32, device=device), members,
                                   noise=None if noise is None else noise[i:i + batch]).cpu().numpy())
    result = xr.Dataset({"precipitation": (("time", "sample", "lat", "lon"), np.concatenate(outputs))},
                        coords={"time": ds.time, "sample": np.arange(members), "lat": ds.lat, "lon": ds.lon})
    result.precipitation.attrs = {"units": "mm", "interval": "(time-1h,time]"}
    result.attrs = {"model": str(path), "target_source": checkpoint["data_attrs"].get("target_source", "unknown"),
                    "temporal_context": cfg.temporal_context,
                    "architecture": cfg.architecture,
                    "terrain": cfg.terrain,
                    "noise_mode": "continuous_anchors" if cfg.temporal_context or cfg.architecture == "joint" else "independent_windows"}
    return result
