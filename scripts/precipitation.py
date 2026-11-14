"""Fetch, train and apply the hourly France rainfall prototype.

Run with ``uv run python scripts/precipitation.py --help``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

from downscale.hourly import default_device
from downscale.precip_data import PARDIG, build_pairs, condition_on_intervals, crop, forecast_pack, plan_pardig
from downscale.precipitation import RainConfig, RainNet, check_pack, evaluate_rain, features, predict, rain_loss, save_model, score


def write_pack(ds, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    # Source coordinates can inherit v3 serializers that are invalid for v2.
    ds = ds.copy(deep=False)
    for variable in ds.variables:
        ds[variable].encoding = {}
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"unfinished write exists: {temporary}")
    ds.to_zarr(temporary, mode="w", zarr_format=2)
    temporary.rename(path)


def train(ds, cfg, out, device):
    check_pack(ds)
    cut = np.datetime64(cfg["validation_start"])
    window = int(ds.attrs["forecast_interval_hours"])
    if len(ds.time) % window or not np.all(np.diff(ds.time.values.reshape(-1, window), axis=1) == np.timedelta64(1, "h")):
        raise ValueError("training pack must contain complete, internally consecutive forecast windows")
    training = np.flatnonzero(ds.time.values < cut)
    validation = np.flatnonzero(ds.time.values >= cut)
    if not len(training) or not len(validation) or len(training) % window or len(validation) % window:
        raise ValueError("validation_start must split complete windows into nonempty train/validation sets")
    settings = cfg["training"]
    model_cfg = RainConfig(**cfg["model"], window_hours=window)
    x_train, mean, std = features(ds.isel(time=training), temporal_context=model_cfg.temporal_context)
    x_val, _, _ = features(ds.isel(time=validation), mean, std, model_cfg.temporal_context)
    ds = ds.copy(deep=False)
    ds.attrs = {**ds.attrs, "training_start": str(ds.time.values[training[0]]),
                "training_end": str(ds.time.values[training[-1]]),
                "validation_start": str(ds.time.values[validation[0]]),
                "validation_end": str(ds.time.values[validation[-1]])}
    torch.manual_seed(settings["seed"])
    rng = np.random.default_rng(settings["seed"])
    net = RainNet(x_train.shape[1], model_cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=settings["learning_rate"], weight_decay=1e-4)
    baseline = ds.predictors.values[:, 0].astype("float32")
    if model_cfg.temporal_context:
        from downscale.rain_temporal import hourly_baseline, trajectory_noise
        baseline = np.empty_like(baseline)
        baseline[training] = hourly_baseline(ds.isel(time=training))
        baseline[validation] = hourly_baseline(ds.isel(time=validation))
    target = ds.target.values.astype("float32")
    sequence_hours = window * settings.get("sequence_windows", 1)
    candidates = [i for i in range(0, len(training) - sequence_hours + 1, window)
                  if np.all(np.diff(ds.time.values[training[i:i + sequence_hours]]) == np.timedelta64(1, "h"))]
    if not candidates:
        raise ValueError("no contiguous training sequences of the configured length")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    if settings["epochs"] < 1 or settings["batch_windows"] < 1:
        raise ValueError("positive epochs and batch_windows required")
    for epoch in range(settings["epochs"]):
        net.train()
        losses = []
        windows = rng.permutation(candidates)
        for step, i in enumerate(range(0, len(windows), settings["batch_windows"])):
            if settings.get("max_steps") is not None and step >= settings["max_steps"]:
                break
            idx = (windows[i:i + settings["batch_windows"], None] + np.arange(sequence_hours)).ravel()
            opt.zero_grad()
            noise = None
            if model_cfg.temporal_context:
                noise = trajectory_noise(ds.time.values[training[idx]], model_cfg.members, len(ds.lat), len(ds.lon), model_cfg.noise_factor, window)
            p = net(torch.tensor(x_train[idx], device=device), torch.tensor(baseline[training[idx]], device=device), noise=noise)
            loss = rain_loss(p, torch.tensor(target[training[idx]], device=device), window,
                             sequence_hours, settings.get("temporal_weight", 0))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if not losses:
            raise ValueError("no optimizer steps; max_steps must be positive")
        net.eval()
        preds = []
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(settings["seed"] + 1)
            val_noise = None
            if model_cfg.temporal_context:
                val_noise = trajectory_noise(ds.time.values[validation], 8, len(ds.lat), len(ds.lon), model_cfg.noise_factor, window)
            for i in range(0, len(validation), window):
                ix = validation[i:i + window]
                preds.append(net(torch.tensor(x_val[i:i + window], device=device), torch.tensor(baseline[ix], device=device), 8,
                                 noise=None if val_noise is None else val_noise[i:i + window]).cpu().numpy())
        pred = np.concatenate(preds)
        metrics = score(pred, target[validation])
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_crps_mm": metrics["crps_mm"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["crps_mm"] < best:
            best = metrics["crps_mm"]
            save_model(out / "model.pt", net, ds, mean, std)
            report = {"reference": "PARDIG; target agreement only, not independent radar validation",
                      "train_hours": len(training), "validation_hours": len(validation),
                      "validation_start": str(ds.time.values[validation[0]]), "config": cfg,
                      "model": metrics, "baseline": score(ds.predictors.values[validation, :1], target[validation]),
                      "hourly_reference_baseline": score(baseline[validation, None], target[validation])}
            # Coherent accumulated samples, retaining the member identity across hours.
            p6 = pred.reshape(-1, window, *pred.shape[1:]).sum(1)
            t6 = target[validation].reshape(-1, window, *target.shape[1:]).sum(1)
            b6 = ds.predictors.values[validation, 0].reshape(-1, window, *target.shape[1:]).sum(1)[:, None]
            report["window_accumulation"] = {"hours": window, "model": score(p6, t6), "baseline": score(b6, t6)}
            (out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return out / "model.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_precipitation.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download hourly ERA5/PARDIG training pairs")
    fetch.add_argument("--start", required=True, help="first interval END, aligned to 01/07/13/19 UTC for 6h forcing")
    fetch.add_argument("--end", required=True, help="last interval END (inclusive)")
    fetch.add_argument("--out", required=True)
    fetch.add_argument("--backend", choices=["arco", "earth2studio"], default="arco")
    fetch.add_argument("--pardig-layout", choices=["auto", "spatial", "timeseries"], default="auto")
    fetch.add_argument("--plan", action="store_true", help="inspect chunk metadata only; fetch no rainfall/predictor fields")
    fit = sub.add_parser("train")
    fit.add_argument("--pairs", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--device", default=default_device())
    fit.add_argument("--epochs", type=int)
    fit.add_argument("--validation-start")
    fit.add_argument("--max-steps", type=int)
    apply = sub.add_parser("predict", help="apply to an already prepared forcing pack")
    apply.add_argument("--model", required=True)
    apply.add_argument("--forcing", required=True)
    apply.add_argument("--out", required=True)
    apply.add_argument("--members", type=int, default=8)
    apply.add_argument("--device", default=default_device())
    evaluate = sub.add_parser("evaluate", help="evaluate an untouched target period")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--pairs", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--members", type=int, default=8)
    evaluate.add_argument("--device", default=default_device())
    forcing = sub.add_parser("forcing", help="convert one global forecast initialization/member to hourly inputs")
    forcing.add_argument("--forecast", required=True, help="Zarr dataset with 'fields' in Earth2Studio layout")
    forcing.add_argument("--model", required=True)
    forcing.add_argument("--accumulation", choices=["cumulative", "interval"], required=True)
    forcing.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.command == "fetch":
        times = pd.date_range(args.start, args.end, freq="h")
        window = cfg["forecast_interval_hours"]
        # Validate bounds before starting a potentially expensive global-chunk read.
        if not len(times) or len(times) % window or (times[0].hour - 1) % window or not times.equals(times.floor("h")) or times.tz is not None:
            raise ValueError("choose complete forcing windows, e.g. 01 UTC through next-day 00 UTC")
        plan = plan_pardig(times, tuple(cfg["region"]), args.pardig_layout)
        print(json.dumps(plan, indent=2), flush=True)
        if args.plan:
            return
        if Path(args.out).exists():
            raise FileExistsError(f"refusing to overwrite {args.out}")
        source = plan["candidates"][plan["chosen"]]["url"]
        # One regional target read for the request avoids re-reading chunks at
        # six-hour boundaries (or full-decade tile chunks in the other layout).
        truth = PARDIG(source, tuple(cfg["region"]))(times, ["tp"]).sel(variable="tp", drop=True)
        batches = []
        for i in range(0, len(times), window):
            print(f"fetching {times[i]} through {times[i + window - 1]}", flush=True)
            ds = build_pairs(times[i:i + window], cfg["predictors"], args.backend, pardig_url=source,
                             box=tuple(cfg["region"]), target=truth, forcing_hours=window)
            ds = condition_on_intervals(ds.load(), window)
            check_pack(ds)
            batches.append(ds)
        write_pack(xr.concat(batches, dim="time"), args.out)
    elif args.command == "train":
        if args.epochs is not None:
            cfg["training"]["epochs"] = args.epochs
        if args.max_steps is not None:
            cfg["training"]["max_steps"] = args.max_steps
        if args.validation_start:
            cfg["validation_start"] = args.validation_start
        train(xr.open_zarr(args.pairs).load(), cfg, args.out, args.device)
    elif args.command == "forcing":
        checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
        fields = crop(xr.open_zarr(args.forecast).fields, tuple(cfg["region"]))
        fields = fields.sel(variable=[v for v in checkpoint["variables"] if v != "phase"])
        ds = forecast_pack(fields, checkpoint["lat"].numpy(), checkpoint["lon"].numpy(),
                           checkpoint["config"]["window_hours"], args.accumulation)
        check_pack(ds, target=False)
        write_pack(ds, args.out)
    elif args.command == "evaluate":
        ds = xr.open_zarr(args.pairs).load()
        checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
        attrs = checkpoint["data_attrs"]
        if not all(f"{split}_{bound}" in attrs for split in ("training", "validation") for bound in ("start", "end")):
            raise ValueError("checkpoint lacks split boundaries; retrain before claiming an untouched test")
        for split in ("training", "validation"):
            if f"{split}_start" in attrs:
                overlap = (ds.time.values >= np.datetime64(attrs[f"{split}_start"])) & (ds.time.values <= np.datetime64(attrs[f"{split}_end"]))
                if overlap.any():
                    raise ValueError(f"test period overlaps {split}; choose an untouched period")
        torch.manual_seed(cfg["training"]["seed"] + 2)
        predictions = predict(args.model, ds, args.device, args.members)
        report = evaluate_rain(predictions, ds)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        write_pack(predictions, out / "predictions.zarr")
        (out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"hourly_model_crps_mm": report["hourly"]["model"]["crps_mm"],
                          "hourly_baseline_crps_mm": report["hourly"]["baseline"]["crps_mm"]}), flush=True)
    else:
        ds = xr.open_zarr(args.forcing).load()
        write_pack(predict(args.model, ds, args.device, args.members), args.out)


if __name__ == "__main__":
    main()
