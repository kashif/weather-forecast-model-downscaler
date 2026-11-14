"""Rainfall units, timing, forecast information, probability scores and training."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from downscale.precip_data import PARDIG, condition_on_intervals, crop, forecast_pack, hourly_pardig, read_cost
from downscale.precipitation import RainConfig, RainNet, check_pack, evaluate_rain, features, predict, rain_loss, save_model, score


def pack(n=12, h=9, w=11):
    rng = np.random.default_rng(4)
    rain = rng.exponential(1, (n, h, w)).astype("float32")
    return xr.Dataset({
        "predictors": (("time", "variable", "lat", "lon"), np.stack([rain, rain + 280], 1)),
        "target": (("time", "lat", "lon"), rain * 1.2),
    }, coords={"time": pd.date_range("2020-10-02T01:00", periods=n, freq="h"),
               "variable": ["tp", "t2m"], "lat": np.linspace(41, 51, h), "lon": np.linspace(-5, 9, w)},
        attrs={"precipitation_units": "mm", "interval": "(time-1h,time]", "target_source": "synthetic"})


def test_pardig_uses_right_labelled_rates_and_preserves_missing_pixels():
    stamps = pd.date_range("2020-01-01", periods=5, freq="15min")
    rate = xr.DataArray(np.arange(5, dtype="float32")[:, None, None] / 1000,
                        dims=("time", "lat", "lon"), coords={"time": stamps, "lat": [45.], "lon": [2.]},
                        attrs={"units": "m h**-1"})
    # Hour ends at 01 UTC; includes .001,.002,.003,.004, excludes 00 UTC.
    result = hourly_pardig(rate, [stamps[-1]])
    assert result.item() == pytest.approx(0.0025)
    assert result.attrs["units"] == "m"
    with pytest.raises(KeyError):
        hourly_pardig(rate.isel(time=[0, 1, 3, 4]), [stamps[-1]])
    rate.values[2] = np.nan
    assert np.isnan(hourly_pardig(rate, [stamps[-1]]).item())
    with pytest.raises(ValueError, match="m/h"):
        hourly_pardig(rate.assign_attrs(units="mm/h"), [stamps[-1]])


def test_crop_crosses_greenwich_without_taking_the_rest_of_the_globe():
    ds = xr.Dataset(coords={"latitude": [46., 45., 44.], "longitude": [0., 1., 2., 358., 359.]})
    selected = crop(ds, (44, 46, -2, 2))
    np.testing.assert_array_equal(selected.lon, [-2, -1, 0, 1, 2])
    np.testing.assert_array_equal(selected.lat, [44, 45, 46])


def test_local_pardig_source_follows_earth2studio_datasource_protocol():
    times = pd.date_range("2020-01-01T00:15", periods=4, freq="15min")
    ds = xr.Dataset({"precipitation": (("time", "latitude", "longitude"), np.full((4, 3, 3), .002))},
                    coords={"time": times, "latitude": [44., 45., 46.], "longitude": [0., 1., 359.]})
    ds.precipitation.attrs["units"] = "m h**-1"
    source = PARDIG(dataset=ds, box=(44, 46, -1, 1))
    result = source("2020-01-01T01:00", "tp")
    assert result.dims == ("time", "variable", "lat", "lon")
    np.testing.assert_allclose(result, .002)
    with pytest.raises(ValueError, match="only"):
        source(times, "t2m")


def test_read_planner_counts_boundary_chunks_without_reading_rainfall():
    ds = xr.Dataset({"precipitation": (("time", "latitude", "longitude"), np.zeros((9, 3, 3), dtype="float32"))},
                    coords={"time": pd.date_range("2020-01-01", periods=9, freq="15min"),
                            "latitude": [44., 45., 46.], "longitude": [0., 1., 359.]})
    ds.precipitation.encoding["chunks"] = (4, 3, 3)
    plan = read_cost(ds.precipitation, ["2020-01-01T01:00"], (44, 46, -1, 1))
    assert plan["chunks_touched"] == 2
    assert plan["decoded_bytes_upper_bound"] == 8 * 3 * 3 * 4
    with pytest.raises(ValueError, match="outside"):
        read_cost(ds.precipitation, ["2020-01-02T01:00"], (44, 46, -1, 1))


def test_endpoint_only_download_matches_full_hourly_conditioning(monkeypatch):
    import downscale.precip_data as data
    ds = pack()
    calls = []

    def fake_era5(times, variables, backend, box):
        calls.append((len(times), list(variables)))
        return ds.predictors.sel(time=times, variable=variables)

    monkeypatch.setattr(data, "era5_fields", fake_era5)
    full = data.condition_on_intervals(data.build_pairs(ds.time.values, ["tp", "t2m"], target=ds.target), 6)
    calls.clear()
    minimal = data.condition_on_intervals(data.build_pairs(ds.time.values, ["tp", "t2m"], target=ds.target, forcing_hours=6), 6)
    xr.testing.assert_equal(full, minimal)
    assert calls == [(12, ["tp"]), (2, ["t2m"])]


def test_evaluation_scores_hourly_and_daily_totals_and_rejects_misalignment():
    ds = condition_on_intervals(pack(n=24), 6)
    pred = xr.Dataset({"precipitation": (("time", "sample", "lat", "lon"), np.repeat(ds.target.values[:, None], 2, axis=1))},
                      coords={"time": ds.time, "sample": [0, 1], "lat": ds.lat, "lon": ds.lon})
    pred.precipitation.attrs["units"] = "mm"
    report = evaluate_rain(pred, ds)
    assert report["hourly"]["model"]["crps_mm"] == 0
    assert report["accumulations"]["24"]["model"]["crps_mm"] == 0
    assert len(report["per_window"]) == 4
    with pytest.raises(ValueError, match="time"):
        evaluate_rain(pred.assign_coords(time=ds.time + np.timedelta64(1, "h")), ds)


def test_pack_writer_removes_remote_v3_encoding_and_refuses_overwrite(tmp_path):
    from scripts.precipitation import write_pack
    ds = pack()
    ds.lat.encoding["serializer"] = {"name": "bytes"}
    path = tmp_path / "pack.zarr"
    write_pack(ds, path)
    np.testing.assert_array_equal(xr.open_zarr(path).lat, ds.lat)
    with pytest.raises(FileExistsError):
        write_pack(ds, path)


def test_interval_conditioning_hides_hourly_rain_and_intermediate_states():
    ds = pack()
    result = condition_on_intervals(ds, 6)
    check_pack(result)
    np.testing.assert_allclose(result.predictors.sel(variable="tp").values[:6].sum(0), ds.predictors.sel(variable="tp").values[:6].sum(0))
    np.testing.assert_allclose(result.predictors.sel(variable="t2m").values[0], ds.predictors.sel(variable="t2m").values[5])
    np.testing.assert_allclose(result.predictors.sel(variable="phase").values[:6, 0, 0], np.arange(1, 7) / 6)
    with pytest.raises(ValueError, match="divide"):
        condition_on_intervals(ds.isel(time=slice(1, None)), 6)


def test_global_forecast_deaccumulates_within_one_run_and_outputs_every_hour():
    ds = pack()
    fields = xr.DataArray(np.ones((3, 2, 9, 11), dtype="float32"),
                          dims=("lead_time", "variable", "lat", "lon"),
                          coords={"lead_time": np.array([0, 6, 12], dtype="timedelta64[h]"),
                                  "variable": ["tp", "t2m"], "lat": ds.lat, "lon": ds.lon,
                                  "time": np.datetime64("2020-10-02T00:00")})
    fields.loc[dict(variable="tp")] = np.array([0, .006, .018])[:, None, None]
    result = forecast_pack(fields, ds.lat.values, ds.lon.values)
    check_pack(result, target=False)
    assert result.sizes["time"] == 12
    np.testing.assert_allclose(result.predictors.sel(variable="tp").values[:6], 1)
    np.testing.assert_allclose(result.predictors.sel(variable="tp").values[6:], 2)
    np.testing.assert_array_equal(result.time, ds.time)
    fields.loc[dict(variable="tp", lead_time=np.timedelta64(12, "h"))] = 0
    with pytest.raises(ValueError, match="reset"):
        forecast_pack(fields, ds.lat.values, ds.lon.values)


def test_network_trains_nonnegative_coherent_samples_and_roundtrips(tmp_path):
    torch.manual_seed(0)
    ds = condition_on_intervals(pack(), 6)
    x, mean, std = features(ds)
    net = RainNet(x.shape[1], RainConfig(base=4, depth=2, window_hours=6))
    captured = []
    handle = net.unet.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    pred = net(torch.tensor(x[:6]), torch.tensor(ds.predictors.values[:6, 0]))
    handle.remove()
    assert pred.shape == (6, 4, 9, 11)
    assert (pred >= 0).all()
    # The same member uses the same spatial noise throughout a six-hour window.
    noise = captured[0][:, -1].reshape(6, 4, *captured[0].shape[-2:])
    torch.testing.assert_close(noise[0], noise[5])
    loss = rain_loss(pred, torch.tensor(ds.target.values[:6]), 6)
    loss.backward()
    assert net.unet.out.weight.grad.abs().sum() > 0
    model = tmp_path / "model.pt"
    save_model(model, net, ds, mean, std)
    torch.manual_seed(12)
    output = predict(model, ds, members=3, batch=6)
    assert output.precipitation.shape == (12, 3, 9, 11)
    assert output.precipitation.attrs["units"] == "mm"
    with pytest.raises(ValueError, match="lon"):
        predict(model, ds.assign_coords(lon=ds.lon + .01))
    with pytest.raises(ValueError, match="forecast_interval"):
        predict(model, ds.assign_attrs(forecast_interval_hours=3))


def test_scores_report_probabilities_neighbourhoods_and_no_event_support():
    truth = np.ones((2, 9, 11), dtype="float32")
    pred = np.repeat(truth[:, None], 4, axis=1)
    metrics = score(pred, truth)
    assert metrics["crps_mm"] == 0
    assert metrics["ensemble_crps_mm"] == 0
    assert metrics["brier_1mm"] == 0
    assert metrics["csi_1mm"] == 1
    assert metrics["fss_1mm_3px"] == 1
    assert metrics["csi_10mm"] is None
    assert metrics["observed_events_10mm"] == 0
    assert sum(b["n"] for b in metrics["reliability_1mm"]) == truth.size
    two = score(np.array([[[[0.]], [[2.]]]]), np.ones((1, 1, 1)))
    assert two["crps_mm"] == 0
    assert two["ensemble_crps_mm"] == .5


def test_cli_runs_training_and_prediction_without_network(tmp_path):
    root = Path(__file__).resolve().parents[1]
    ds = condition_on_intervals(pack(), 6)
    pairs = tmp_path / "pairs.zarr"
    ds.to_zarr(pairs, zarr_format=2)
    run = tmp_path / "run"
    subprocess.run([sys.executable, str(root / "scripts/precipitation.py"), "train", "--pairs", str(pairs),
                    "--out", str(run), "--device", "cpu", "--epochs", "1", "--max-steps", "1",
                    "--validation-start", "2020-10-02T07:00"], cwd=root, check=True, capture_output=True)
    report = json.loads((run / "evaluation.json").read_text())
    assert report["train_hours"] == report["validation_hours"] == 6
    assert np.isfinite(report["model"]["crps_mm"])
    assert report["window_accumulation"]["hours"] == 6
    output = tmp_path / "rain.zarr"
    subprocess.run([sys.executable, str(root / "scripts/precipitation.py"), "predict", "--model", str(run / "model.pt"),
                    "--forcing", str(pairs), "--out", str(output), "--device", "cpu"], cwd=root, check=True, capture_output=True)
    assert xr.open_zarr(output).precipitation.shape == (12, 8, 9, 11)
    rejected = subprocess.run([sys.executable, str(root / "scripts/precipitation.py"), "evaluate", "--model", str(run / "model.pt"),
                               "--pairs", str(pairs), "--out", str(tmp_path / "invalid_test")], cwd=root, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "test period overlaps training" in rejected.stderr
