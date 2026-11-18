"""Fit separate French gauge event probabilities from cached station forecasts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from downscale.rain_alerts import contingency, paired_day_csi_interval, precision_recall
from downscale.rain_resume import atomic_json
from scripts.check_rain_gauges import aligned_observations, paired_brier_interval, quality_control


ROOT = Path('data/precipitation')
FPR_BUDGET = 0.001
THRESHOLDS = [5., 10.]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(run, gauge_cache, dates=None, method='nearest'):
    manifest = json.loads((run / 'manifest.json').read_text())
    config = manifest['config']
    available = config['splits']['test']
    dates = available if dates is None else dates
    observation_path = gauge_cache / 'observations.parquet'
    if digest(observation_path) != manifest['observations']:
        raise ValueError('Gauge observations differ from the forecast manifest')
    stamps = np.concatenate([pd.date_range(date, periods=24, freq='h').values
                             + np.timedelta64(1, 'h') for date in available])
    rows, stations, _ = quality_control(pd.read_parquet(observation_path), stamps,
                                        config['region'], config['gauge_min_completeness'])
    if stations.index.tolist() != manifest['stations']:
        raise ValueError('Gauge station order differs from cached forecasts')
    cases = []
    for date in dates:
        if date not in available:
            raise ValueError(f'Date {date} is missing from the forecast manifest')
        times = pd.date_range(date, periods=24, freq='h') + pd.Timedelta(hours=1)
        obs = aligned_observations(rows, stations, times)
        valid = np.isfinite(obs)
        with np.load(run / f'{date}.npz') as fields:
            raw = fields[f'{method}_raw_prob'][valid]
            exp = fields[f'{method}_exponential_prob'][valid]
            mean = fields[f'{method}_raw_mean'][valid]
        cases.append((date, obs[valid], raw, exp, mean))
    return cases, manifest


def unpack(cases, index):
    truth = np.concatenate([case[1] for case in cases]) > THRESHOLDS[index]
    raw = np.concatenate([case[2][:, index] for case in cases])
    exp = np.concatenate([case[3][:, index] for case in cases])
    mean = np.concatenate([case[4] for case in cases])
    features = np.column_stack([logit(np.clip(exp, 1e-8, 1-1e-8)), raw, np.log1p(mean)])
    return truth, exp, features


def fit_logistic(features, observed):
    mean = features.mean(0)
    scale = features.std(0).clip(1e-8)
    x = np.column_stack([np.ones(len(features)), (features - mean) / scale])
    prevalence = observed.mean()
    if not 0 < prevalence < 1:
        raise ValueError('Both events and non-events are required')
    penalty = 1e-5

    def objective(beta):
        logits = x @ beta
        loss = np.mean(np.logaddexp(0, logits) - observed * logits)
        gradient = x.T @ (expit(logits) - observed) / len(x)
        gradient[1:] += penalty * beta[1:]
        return loss + penalty * np.sum(beta[1:] ** 2) / 2, gradient

    initial = np.r_[logit(prevalence), np.zeros(features.shape[1])]
    result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                      options={'gtol': 1e-9, 'ftol': 1e-13})
    if not result.success:
        raise RuntimeError(result.message)
    return dict(mean=mean.tolist(), scale=scale.tolist(), coefficients=result.x.tolist())


def predict(model, features):
    beta = np.asarray(model['coefficients'])
    return expit(beta[0] + ((features - model['mean']) / model['scale']) @ beta[1:])


def cutoff_for_budget(probability, observed, budget=FPR_BUDGET):
    """Use the lowest cutoff that excludes enough negatives, including ties."""
    negative = probability[~observed]
    if not len(negative):
        raise ValueError('Cutoff selection requires non-events')
    index = len(negative) - int(np.floor(budget * len(negative))) - 1
    cutoff = float(np.nextafter(np.partition(negative, index)[index], np.inf))
    return cutoff if cutoff <= 1 else None


def model_predictions(model, cases, index):
    observed, exponential, features = unpack(cases, index)
    return observed, {
        'exponential': exponential,
        'gauge_scale': predict(model['gauge_scale'], features[:, :1]),
        'gauge_blend': predict(model['gauge_blend'], features),
    }


def score(model, cases, index):
    observed, predictions = model_predictions(model, cases, index)
    day = np.repeat(np.arange(len(cases)), [len(case[1]) for case in cases])
    report = {}
    for name, probability in predictions.items():
        curve = precision_recall(probability, observed, FPR_BUDGET)
        cutoff = model['cutoffs'][name]
        report[name] = dict(
            average_precision=curve['average_precision'],
            test_curve_recall_at_fpr=curve['recall_at_fpr_curve_diagnostic'],
            brier=float(np.mean((probability - observed) ** 2)),
            mean_probability=float(probability.mean()),
            frozen_alert=contingency(probability, observed, cutoff),
            per_day=[dict(date=case[0], **contingency(probability[day == i], observed[day == i], cutoff))
                     for i, case in enumerate(cases)],
        )
    baseline = report['exponential']
    for name in ['gauge_scale', 'gauge_blend']:
        row = report[name]
        row['brier_vs_exponential'] = paired_brier_interval(
            predictions['exponential'], predictions[name], observed, day, len(cases))
        row['csi_vs_exponential'] = paired_day_csi_interval(baseline['per_day'], row['per_day'])
        row['csi_vs_exponential']['warning'] = 'Exploratory bootstrap by date'
    return dict(station_hours=len(observed), events=int(observed.sum()),
                prevalence=float(observed.mean()), models=report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('runs/gauge_alert_calibration'))
    parser.add_argument('--fit-only', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frozen_path = args.out / 'frozen.json'
    development = Path('runs/rain_gauges')
    fit_dates = [f'2025-{month:02d}-05' for month in range(1, 13)]
    selection_dates = [f'2025-{month:02d}-20' for month in range(1, 13)]
    if not frozen_path.exists():
        fit, manifest = load_cases(development, ROOT / 'gauge_validation', fit_dates)
        validation, _ = load_cases(development, ROOT / 'gauge_validation', selection_dates)
        frozen = dict(fit_dates=fit_dates, selection_dates=selection_dates,
                      threshold_mm=THRESHOLDS, false_positive_budget=FPR_BUDGET,
                      downscaler=manifest['downscaler'], calibrators=manifest['calibrators'],
                      fit_manifest_sha256=digest(development / 'manifest.json'), models={})
        for index, threshold in enumerate(THRESHOLDS):
            truth, _, features = unpack(fit, index)
            model = dict(gauge_scale=fit_logistic(features[:, :1], truth),
                         gauge_blend=fit_logistic(features, truth))
            observed, predictions = model_predictions(model, validation, index)
            model['cutoffs'] = {name: cutoff_for_budget(prob, observed)
                                for name, prob in predictions.items()}
            model['validation'] = score(model, validation, index)
            frozen['models'][str(threshold)] = model
        atomic_json(frozen_path, frozen)
    if args.fit_only:
        return
    frozen = json.loads(frozen_path.read_text())
    test_run = Path('runs/rain_gauge_calibration_test')
    test_manifest = json.loads((test_run / 'manifest.json').read_text())
    dates = test_manifest['config']['splits']['test']
    if set(dates) & set(frozen['fit_dates'] + frozen['selection_dates']):
        raise ValueError('Gauge test overlaps development dates')
    for name in ['downscaler', 'calibrators']:
        if test_manifest[name] != frozen[name]:
            raise ValueError(f'Gauge test {name} differs from fitting')
    report = dict(test_dates=dates, frozen_sha256=digest(frozen_path),
                  test_manifest_sha256=digest(test_run / 'manifest.json'), evaluations={})
    for method in ['nearest', 'bilinear']:
        cases, _ = load_cases(test_run, ROOT / 'gauge_calibration_test', method=method)
        report['evaluations'][method] = {
            str(t): score(frozen['models'][str(t)], cases, index)
            for index, t in enumerate(THRESHOLDS)
        }
    atomic_json(args.out / 'results.json', report)


if __name__ == '__main__':
    main()
