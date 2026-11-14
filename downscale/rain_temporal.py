"""Neighbouring forecast context and continuous stochastic rainfall trajectories."""
from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator


def segments(times, hours):
    """Contiguous forecast segments; never connect separate events or split gaps."""
    times = np.asarray(times, dtype="datetime64[ns]")
    if not len(times) or len(times) % hours:
        raise ValueError("complete forecast windows required")
    windows = times.reshape(-1, hours)
    if not np.all(np.diff(windows, axis=1) == np.timedelta64(1, "h")):
        raise ValueError("hours within each forecast window must be consecutive")
    cuts = np.flatnonzero(windows[1:, 0] - windows[:-1, -1] != np.timedelta64(1, "h")) + 1
    edges = [0, *cuts.tolist(), len(windows)]
    return [slice(a * hours, b * hours) for a, b in zip(edges[:-1], edges[1:])]


def hourly_baseline(ds):
    """Differentiate monotone cumulative-rain interpolation into hourly depths.

    PCHIP has continuous first derivatives, preserves monotonicity and passes
    through every accumulation endpoint. Hourly depths remain nonnegative and
    sum to each original six-hour total. No rainfall observations are used.
    """
    hours = int(ds.attrs["forecast_interval_hours"])
    rain = ds.predictors.sel(variable="tp").values.astype("float64")
    result = np.empty_like(rain, dtype="float32")
    for part in segments(ds.time.values, hours):
        values = rain[part]
        totals = values.reshape(-1, hours, *rain.shape[1:]).sum(1)
        cumulative = np.concatenate([np.zeros_like(totals[:1]), totals.cumsum(0)])
        interpolator = PchipInterpolator(np.arange(len(cumulative)) * hours, cumulative, axis=0, extrapolate=False)
        result[part] = np.maximum(np.diff(interpolator(np.arange(len(values) + 1)), axis=0), 0)
    return result


def context_inputs(ds):
    """Continuous forcing paths plus six-hour look-behind/look-ahead.

    No sawtooth/cyclic window phase or piecewise-constant window-role channels:
    those give an hourly decoder a shortcut to learn six-hour pulses. All paths
    come from the issued forecast, never from future observed targets.
    """
    hours = int(ds.attrs["forecast_interval_hours"])
    names = [str(v) for v in ds.variable.values if v != "phase"]
    raw = ds.predictors.sel(variable=names).values.astype("float32")
    windows = raw.reshape(-1, hours, *raw.shape[1:])[:, 0]
    main = np.empty_like(raw)
    main[:, 0] = hourly_baseline(ds)
    before, after = np.empty_like(raw), np.empty_like(raw)
    past_weight, future_weight = np.empty(len(raw), dtype="float32"), np.empty(len(raw), dtype="float32")
    for part in segments(ds.time.values, hours):
        a, b = part.start // hours, part.stop // hours
        length = part.stop - part.start
        if len(names) > 1:
            # Endpoint atmospheric states. The initial endpoint is unavailable
            # in legacy packs, so hold the first state over that edge interval.
            states = np.concatenate([windows[a:a + 1, 1:], windows[a:b, 1:]])
            path = PchipInterpolator(np.arange(len(states)) * hours, states, axis=0, extrapolate=False)
            main[part, 1:] = path(np.arange(1, length + 1))
        index = np.arange(length)
        before[part] = main[part][np.maximum(index - hours, 0)]
        after[part] = main[part][np.minimum(index + hours, length - 1)]
        past_weight[part] = np.clip((index + .5) / hours, 0, 1)
        future_weight[part] = np.clip((length - index - .5) / hours, 0, 1)
    shape = (len(raw), 1, *raw.shape[-2:])
    extras = [np.broadcast_to(past_weight[:, None, None, None], shape),
              np.broadcast_to(future_weight[:, None, None, None], shape)]
    values = np.concatenate([main, *extras, before, after], axis=1).astype("float32")
    feature_names = [*names, "past_context_weight", "future_context_weight",
                     *[f"previous_{v}" for v in names], *[f"next_{v}" for v in names]]
    return values, feature_names


def trajectory_noise(times, members, height, width, factor=8, hours=6):
    """Member noise interpolated across forecast boundaries, independent of batching.

    Low-resolution Gaussian anchors are drawn at window boundaries. Linear
    interpolation with variance normalization produces a continuous latent path;
    gaps start independent paths. Call once before slicing inference batches.
    """
    nh, nw = max(1, height // factor), max(1, width // factor)
    result = torch.empty(len(times), members, 1, nh, nw)
    for part in segments(times, hours):
        length = part.stop - part.start
        anchors = torch.randn(length // hours + 1, members, 1, nh, nw)
        position = (torch.arange(length, dtype=torch.float32) + .5) / hours
        left = position.floor().long()
        weight = (position - left)[:, None, None, None, None]
        result[part] = (anchors[left] * (1 - weight) + anchors[left + 1] * weight) / torch.sqrt((1 - weight) ** 2 + weight ** 2)
    return result
