# weather-forecast-model-downscaler

The experimental hourly **France precipitation downscaler** is described below under
[Hourly precipitation](#hourly-precipitation). Temperature workflows remain as documented here.

Train a downscaler that turns a 0.25° ensemble temperature forecast into 5.5 km maps over France, and
apply it to AIFS Ensemble 1.0 forecasts read from Earthmover.

The downscaler is a small U-Net trained on pairs of ERA5 (coarse) and CERRA (fine). It learns the
correction from the interpolated coarse field to the fine one, from the terrain, the coast, the time of
day and of year, and a draw of noise, so every coarse member yields several plausible fine maps.
Running the forecast models themselves lives in a separate project,
[earth2studio-pipeline](https://github.com/EmmaScharfmann/earth2studio-pipeline).

## Setup

```bash
uv sync
uv run arraylake auth login      # Earthmover account: ERA5 and AIFS-ENS
uv run hf auth login             # model and datasets on the Hugging Face Hub
```

AIFS-ENS comes from the [Earthmover marketplace](https://app.earthmover.io/marketplace/6a10729e8a69e6bcaeae0f90).
Subscribing creates a repository under your organisation; set `forecasts.source` in `config_spatial.yaml` to it.

## Tutorial

`tutorial_inference.ipynb` fetches an AIFS-ENS run, applies the published downscaler
([EmmaScharfmann/france-temperature-downscaler](https://huggingface.co/EmmaScharfmann/france-temperature-downscaler)),
maps both forecasts, and evaluates them against CERRA, CERRA's own floor and Météo-France stations.

`training_tutorial.ipynb` trains the downscaler from scratch: it pairs ERA5 from Earthmover with CERRA
from the Hub, trains on 2014 to 2019, scores the model on 2020 against interpolation, and saves a
checkpoint that `tutorial_inference.ipynb` can load.

`tutorial_temporal.ipynb` fills the hours between AIFS-ENS's 6-hourly steps with the hourly model
([kashif/france-temperature-hourly-downscaler](https://huggingface.co/kashif/france-temperature-hourly-downscaler)),
reading the forecast from ECMWF's open data, and scores it hour by hour against the stations.

## Configuration

`config_spatial.yaml` configures the spatial downscaler and everything shared (region, paths, stations,
evaluation). `config_temporal.yaml` starts with `extends: config_spatial.yaml` and lists only what the
hourly model changes, plus its own `temporal` section, so the two files show what is common and what is
specific. Every script takes `--config`; command-line flags override it.

## Pipeline

| Step | Script |
|---|---|
| Download the CERRA target (Copernicus account) | `scripts/download_cerra.py` |
| Pair it with ERA5 from Earthmover | `scripts/build_pairs.py`, then `--consolidate` |
| Train | `scripts/train_downscaler.py --out runs/crps` |
| Check the model against its target's floor | `scripts/check_target_floor.py --model runs/crps/model.pt --start ... --end ...` |
| Check the power spectrum | `scripts/check_spectrum.py --model runs/crps/model.pt` |
| Fetch AIFS-ENS forecasts | `scripts/fetch_forecasts.py --start 2025-08-01 --end 2025-08-31` |
| Downscale them | `scripts/downscale_forecasts.py --model runs/crps/model.pt` |
| Score both against stations | `scripts/eval_downscaled.py` |
| Publish the model or datasets to the Hub | `scripts/publish.py` |

For hourly output between the 6-hourly steps, every script takes `--config config_temporal.yaml`:

| Step | Script |
|---|---|
| Download hourly CERRA (analyses plus their +1 h/+2 h forecasts) | `scripts/download_cerra.py` |
| Pair it with ERA5, from Google's public copy | `scripts/build_pairs.py` |
| Train, starting from the spatial model | `scripts/train_downscaler.py --temporal --out runs/hourly` |
| Fine-tune it with CRPS on block averages too, against grain from the noise | `scripts/train_downscaler.py --temporal --init-hourly runs/hourly/model.pt --multiscale 2 4 8 16 --epochs 6 --learning-rate 3e-4 --out runs/hourly_ms` |
| Fetch AIFS-ENS from ECMWF open data, no account needed | `scripts/fetch_forecasts.py --start ... --end ...` |
| Downscale every hour, with the spatial model as baseline | `scripts/downscale_hourly.py --hourly runs/hourly_ms/model.pt --spatial runs/spatial/model.pt` |
| Score each against stations | `scripts/eval_downscaled.py --forecasts outputs/hourly/forecasts/*.zarr --downscaled outputs/hourly/hourly/*.zarr` |
| Animate one forecast hour by hour, next to the stations | `scripts/movie_hourly.py --run outputs/hourly --stations` |
| Power spectra in space and time against CERRA | `scripts/check_spectrum.py --model runs/spatial/model.pt --hourly runs/hourly_ms/model.pt` |

The hourly pairs must be on the same grid as the spatial model, which `target.crop_like` takes care of for the
published one; training refuses them otherwise. `scripts/movie_hourly.py --forecast ... --downscaled ...`
animates the spatial model alone, at the forecast's own steps. `fetch_forecasts.py --source open-data:ifs-ens`
reads IFS ENS (or `ifs`, `aifs-single`) from the same open data.

## Layout

- `downscale/`: the library. `sources.py` (ERA5 and AIFS-ENS access), `open_data.py` (ECMWF open data),
  `downscaler.py` (network and checkpoint), `hourly.py` (the hourly model), `inference.py` (apply to a forecast), `evaluation.py` and `plots.py` (scores and figures
  for the tutorial), `stations.py` and `station_scoring.py` (observations and scoring), `diagnostics.py`
  (report figures), `fetch.py` and `publish.py` (Hub datasets).
- `scripts/`: the pipeline steps above.
- `tests/`: `uv run pytest`, on synthetic data, no network needed.
- `findings.md`: behaviour of the data and libraries worth knowing.

## Hourly precipitation

`config_precipitation.yaml` and `scripts/precipitation.py` implement a separate regional
rainfall prototype: **global forecast forcing → hourly probabilistic rainfall over France**.
The initial target is PARDIG-NRT at 0.1° (roughly 8 km east–west × 11 km north–south
over France). The bounding box includes France and surrounding countries and sea;
reported scores cover that box, not a France administrative land mask.

Historical training uses ERA5 at 0.25° with PARDIG targets. For six-hour forcing,
the model receives the interval's average rainfall, its endpoint temperature,
dewpoint, 10 m wind components and pressure, and the hour's phase in the interval.
The intermediate hourly ERA5 states are hidden to match forecast-time inputs.
Geographical coordinates and calendar cycles supply spatial and diurnal context.
This baseline does not yet have a fine-resolution terrain encoder or upper-air inputs.

### Architecture evidence and implementation

The [WN3 paper](https://arxiv.org/abs/2609.03582), especially §§2.2 and A.1,
is more precise than the [announcement](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/introducing-weathernext-3/):

| Component | What the paper describes | What this branch implements |
| --- | --- | --- |
| Global backbone | 0.25° and 0.1° encoders/decoders, shared 40,962-node icosahedral mesh, 32 transformer layers, latent width 1,024 | A regional U-Net with bilinear decoder upsampling; no global backbone retraining |
| Hourly output | Native hourly subtime channels inside six-hour outer steps | Hour-conditioned decoder evaluated for six successive hours, with shared member noise and an accumulation loss; it is not WN3's simultaneous subtime-channel decoder |
| Stochastic predictions | Noise modulates conditional normalization; two samples during CRPS training | Coherent low-resolution spatial noise concatenated to inputs, four training members |
| Rainfall objective | Marginal fair CRPS; additional globally pooled CRPS weighted 0.3; 1 h and 6 h accumulation targets | Physical-unit multiscale fair CRPS, domain-mean CRPS at 0.3, interval accumulation CRPS at 0.3 after division by interval hours |
| Local observation head | Refine 0.1° latents with a 4-layer width-768 CNN; bilinear latent sampling; metadata MLP for elevation, land/sea and time; 2-hidden-layer width-768 output MLP | Deferred: requires independent high-resolution observations; gridded rainfall output only |
| 0.05° output | Station-trained temperature and dewpoint, with backbone frozen during head training | No claim that PARDIG supplies validated 0.05° rainfall |

The regional U-Net predicts an additive correction in log1p rainfall space, then
decodes nonnegative depths. An initial 0.01 offset permits gradients over dry inputs;
the numerical upper bound is 500 mm/h. This is an implementation bound, not an
extreme-rain skill claim. Shared noise and accumulated losses encourage temporal
coherence; outputs are not constrained to exactly reproduce the coarse total, since
the target may correct coarse-model rainfall bias.

The [PARDIG paper](https://arxiv.org/abs/2611.13941), §§2.3–2.4 and A.3.4,
describes a related **retrieval** system, not a causal global forecaster. It uses
four hours of analysis/satellite context, including observations after the target
time, a 16-layer width-512 mesh transformer, and a continuous query decoder trained
with cross-entropy over 14 rainfall bins against sparse GPM CORRA observations.
The public dense product is a calibrated deterministic reconstruction, rather than
an ensemble of precipitation observations. Our CRPS ensemble measures uncertainty
conditional on those proxy targets; matching PARDIG does not certify calibration
against actual rainfall.

### Data conventions and satellite availability

The [PARDIG guide](https://developers.google.com/weathernext/guides/pardig)
and live metadata specify instantaneous `precipitation` in **m/h**. Following
paper Eq. A.3.8, each hour ending at `t` uses rates at `t−45, t−30, t−15, t`
minutes, each weighted by 0.25 hours, then converted to **mm**. Missing samples
raise or remain missing; they never become zero rainfall. ERA5 `tp` is already
the preceding hour's accumulated depth in metres, so it is multiplied by 1,000.
Greenwich-crossing crops are selected before interpolation and sorted into signed
longitude. Interpolation of rainfall is bilinear, not area-conservative.

Use NRT consistently for 2016 onward; do not silently concatenate the LR and NRT
variants. Their upstream inputs differ. PARDIG is a learned reference with known
limitations in the Alps, snowfall and drizzle, and a 75 mm/h instantaneous-rate
cap. The 0.1° target cannot supervise genuine 1–5 km rainfall detail. A later stage
can use radar/gauge targets on a finer grid; independent evaluation must establish
whether the added detail is useful.

The [SatWeave guide](https://developers.google.com/weathernext/guides/satweave)
documents 17 bands, 30-minute archives in 2016–2025, hourly operational imagery,
and approximately one-hour latency. WN3 used an 11-band subset. Meteosat-compatible
thermal/water-vapour bands (e.g. 6.2 and 11.2 μm) are sensible France inputs;
several other bands have no European coverage and visible bands are missing at night.
**This prototype does not ingest SatWeave.** Future imagery at a forecast's valid
time would leak observations. A satellite-assisted extension must use only frames
available at issuance, encode their ages and validity masks, and measure gains by
lead time against a forcing-only ablation. PARDIG is target-only because its
publication latency is approximately 2–8 hours.

### Run a small real-data training check

No Earthmover account is needed for the direct ARCO/PARDIG path. The reader uses
the guide's `obstore.store.GCSStore` → `zarr.storage.ObjectStore` → `xr.open_zarr`
pattern with anonymous public reads and backend-lazy indexing. Public spatial
chunks span the globe: even a France crop downloads global compressed chunks.
`--plan` compares chunk metadata for the spatial and time-series layouts without
fetching rainfall or ERA5 fields; `--pardig-layout auto` chooses the smaller
decoded-byte estimate. This is a chunk-cost estimate, not measured compressed
network traffic. Only the `precipitation` field, requested quarters of each hour,
and France cells are selected; all target hours are read together to avoid
re-fetching shared boundary chunks. No SatWeave bands are downloaded in this version.
Start small; fetch bounds are inclusive hourly interval endpoints.

```bash
uv sync
uv run python scripts/precipitation.py fetch \
  --start 2020-10-02T01:00 --end 2020-10-03T00:00 \
  --out data/precipitation/pilot.zarr --plan
uv run python scripts/precipitation.py fetch \
  --start 2020-10-02T01:00 --end 2020-10-03T00:00 \
  --out data/precipitation/pilot.zarr
uv run python scripts/precipitation.py train \
  --pairs data/precipitation/pilot.zarr --out runs/precipitation_pilot \
  --epochs 2 --max-steps 2 --validation-start 2020-10-02T19:00
uv run python scripts/precipitation.py predict \
  --model runs/precipitation_pilot/model.pt \
  --forcing data/precipitation/pilot.zarr --out outputs/precipitation_pilot.zarr
```

That single-event split is a pipeline smoke test, not an independent skill estimate.
`model.pt`, `history.json` and `evaluation.json` are written under the run directory;
the best checkpoint is selected on validation CRPS. For a scientific experiment,
use complete seasons and split by years/events, for example 2016–2021 training,
2022 validation and 2023–2025 untouched testing. The current trainer loads the
regional pack into RAM: use small pilot packs before building a streaming trainer
for a multi-year experiment. The separate `evaluate` command rejects test dates
overlapping the checkpoint's recorded training or validation periods.

Local pilot completed on Apple MPS: 24 hours on a 131 × 180 grid, 18 training
hours and 6 validation hours, two epochs with two optimizer steps each. Best
validation hourly CRPS was 0.6416 mm versus 0.6635 mm for the interval-mean coarse
baseline; six-hour-total CRPS was 3.0015 mm versus 3.1220 mm. Hourly 1 mm CSI
was slightly worse (0.3726 versus 0.3750), so the metrics do not show uniform
improvement. These are PARDIG agreement numbers on one event, without independent
radar validation or an operational global-forecast test. The metadata-only read
plan for this 24-hour request touches 25 global spatial chunks (2.59 GB decoded)
versus 999 full-time-series tiles (35.03 GB decoded); compressed transfer sizes
are data-dependent.

### Cross-year training, test evaluation and movie

`config_precipitation_experiment.yaml` records the completed small experiment:
96 training hours from 21–24 November 2019, 24 validation hours from 2 October
2020, and 24 untouched test hours from 3 October 2021 (all right-labelled UTC).
The fresh model trained for 30 epochs / 480 optimizer steps on Apple MPS, using
only training statistics for normalization. Epoch 21 was selected on validation
CRPS. The test period was evaluated once with eight samples after selection;
adding finite-ensemble metrics reused the same saved predictions.

| Test metric | Coarse interval mean | Downscaler |
| --- | ---: | ---: |
| Hourly fair CRPS, mm ↓ | 0.4138 | 0.2714 |
| Hourly finite-ensemble CRPS, mm ↓ | 0.4138 | 0.2959 |
| Hourly ensemble-mean RMSE, mm ↓ | 0.9520 | 0.9506 |
| Hourly bias, mm (ideal 0) | +0.0688 | −0.0187 |
| Brier score, ≥1 mm ↓ | 0.1211 | 0.0888 |
| CSI, ≥1 mm ↑ | 0.4416 | 0.3608 |
| FSS, ≥1 mm over 9-pixel neighbourhoods ↑ | 0.7518 | 0.6966 |
| Six-hour fair CRPS, mm ↓ | 1.7962 | 1.2044 |
| Daily fair CRPS, mm ↓ | 5.4290 | 3.7895 |

Probability scores and rainfall-volume bias improve; RMSE barely changes, and
threshold-event footprint scores worsen. This does not establish sharper or
more accurate storm placement. These are comparisons to the learned PARDIG
target with ERA5 forcing, not independent rainfall verification or operational
forecast skill. Four training days and one test day cannot establish seasonal
performance. The [recorded results](examples/france_precipitation/results.json)
include split dates and the checkpoint checksum.

The local Neural-LAM checkout (`../neural-lam`, inspected at `9d5aa0f`) provided
useful checks: keep forcing distinct from prediction targets, normalize each
category exactly once, separate train/validation/test, and inspect regional
boundaries separately. See `neural_lam/models/modules/base.py` and
`neural_lam/models/forecasters/base.py` there. This downscaler retains its U-Net
and does not copy target values onto domain boundaries; test reports additionally
exclude an eight-cell context ring for an interior comparison.

For the cached training/validation pack and separate test pack:

```bash
uv run python scripts/precipitation.py --config config_precipitation_experiment.yaml train \
  --pairs data/precipitation/train_validation.zarr --out runs/precipitation_cross_year
uv run python scripts/precipitation.py --config config_precipitation_experiment.yaml evaluate \
  --model runs/precipitation_cross_year/model.pt \
  --pairs data/precipitation/test_2021.zarr --out outputs/precipitation_cross_year/test
uv run python scripts/movie_precipitation.py \
  --pairs data/precipitation/test_2021.zarr \
  --predictions outputs/precipitation_cross_year/test/predictions.zarr \
  --out outputs/precipitation_cross_year/test/france_hourly_rain.mp4 \
  --gif outputs/precipitation_cross_year/test/france_hourly_rain.gif
```

Fetch each period with `fetch --start ... --end ...` as above; the `periods` entries
in the experiment config record bounds rather than automatically downloading
them. Concatenate the training and validation packs in time using
`xr.concat([train, validation], dim="time", join="exact")`; the trainer permits
gaps between complete six-hour windows. Atmospheric predictors are downloaded
only at the endpoint actually used by the model, reducing five of every six
temperature/dewpoint/wind/pressure reads. Rainfall still requires all hourly
increments to construct the six-hour forcing total.

The MP4/GIF compares coarse interval-mean forcing, one stochastic member,
ensemble mean and PARDIG with common rainfall bins, plus a domain-mean time
series. Rendering an MP4 requires `ffmpeg`; GIF output uses Pillow.
The 24-frame, 12-second test animation is available as a
[GIF preview](examples/france_precipitation/france_hourly_rain.gif).

### Global forecast forcing and Earth2Studio

The direct xarray path works on this project's CPU/Apple MPS environment. The
optional `--backend earth2studio` calls `earth2studio.data.ARCO_ERA5` for historical
inputs when Earth2Studio is installed in your environment. The `PARDIG` class in
`downscale/precip_data.py` follows Earth2Studio's `(time, variable) -> DataArray`
interface for regional hourly `tp` in metres. There is no built-in PARDIG/SatWeave
Earth2Studio adapter assumed here. Earth2Studio is an optional integration, not
required for the direct training path.

At deployment the forcing comes from a **global forecast**, such as
[IFS_FX](https://nvidia.github.io/earth2studio/main/modules/generated/data/forecast/IFS_FX/),
AIFS or a compatible Earth2Studio rollout. Export one initialization and one global
ensemble member at a time as a Zarr dataset with `fields(lead_time, variable, lat,
lon)` and a scalar `time` initialization coordinate. `tp` must be **metres**;
other variables use the training units (K, m/s, Pa). Six-hour leads must be equally
spaced. For accumulations since initialization, include lead zero; for interval
totals, start at lead six. Verify these semantics in the chosen model's metadata;
Earth2Studio variable names alone do not establish an accumulation interval.

```bash
uv run python scripts/precipitation.py forcing \
  --forecast outputs/global_member.zarr --accumulation cumulative \
  --model runs/precipitation_pilot/model.pt --out outputs/france_forcing.zarr
uv run python scripts/precipitation.py predict \
  --forcing outputs/france_forcing.zarr --model runs/precipitation_pilot/model.pt \
  --out outputs/france_hourly_rain.zarr
```

`forecast_pack` deaccumulates within that single run/member, rejects negative
resets, interpolates onto the trained France grid and produces each hourly phase.
Use a halo around France when exporting fields to cover interpolation boundaries.
Apply separately to every global ensemble member, retaining global member IDs
alongside the downscaler sample IDs in downstream workflows. Forecast access and
running a global model remain external to this CLI. The forecast adapter is tested
with synthetic cumulative fields; an operational IFS/AIFS integration has not yet
been validated. Reanalysis-trained downscalers need evaluation and preferably
fine-tuning on forecast hindcasts, stratified by lead time.

### Evaluation recipe from both papers

`evaluation.json` compares the downscaler with the interval-mean/interpolated coarse
baseline on the same held-out PARDIG hours. Scores are uniform pixel averages over
the rectangular context domain; they are not area-weighted or France-land-only.

| Metric | Why it matters | Current implementation |
| --- | --- | --- |
| Fair and finite-ensemble CRPS | Generator score and actual ensemble score in physical rainfall units (WN3 A.2.1 distinguishes these) | Hourly, forcing-interval and daily totals; fair CRPS with ≥2 members |
| RMSE and bias | Mean error and water-volume bias | Hourly and interval totals |
| Brier score and reliability | Rain-event probabilities and calibration | 0.1, 1, 5, 10 mm thresholds, reliability bins and observed-event counts |
| CSI | Hits versus misses and false alarms; both papers evaluate threshold skill | Median-probability event at the same thresholds; null for no-event unions |
| FSS | Tolerates small displacement errors and checks storm footprints (PARDIG A.10.6) | Median-probability footprint, 1/3/9-pixel square neighbourhoods; boundary padding excluded |
| Multi-hour totals | Timing and catchment-relevant rainfall (PARDIG A.13; WN3 A.1.2) | Forcing-interval totals; thresholds denote total mm over that interval |

Before a skill claim, add independent rain gauges and quality-controlled radar
(e.g. EURADCLIM/OPERA where access and rainfall products permit), conservative
aggregation to common spatial support, 24-hour and catchment totals, spectra and
temporal structure checks, equal-area neighbourhoods, land/terrain/season/lead
stratification, and confidence intervals bootstrapped by storm or initialization.
Evaluate interpolation, statistical calibration, the existing U-Net baseline and
the proposed decoder at matched member counts. Reserve post-2023 radar/gauge data
where possible because PARDIG itself was trained on 2016–2023 satellite targets.
PARDIG agreement and independent observation skill must be reported separately.
