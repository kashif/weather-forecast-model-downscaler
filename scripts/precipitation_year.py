"""Resumable annual PARDIG/ERA5 preparation, training and held-out evaluation.

Run from the repository root. Regional tiles are read once across all requested
periods, using the guide's unsigned obstore GCS access. No global arrays persist.
"""
from __future__ import annotations
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import zarr
import obstore
from downscale.precip_data import open_store, crop, ERA5_NAMES, PARDIG_TIMESERIES, condition_on_intervals
from downscale.sources import ERA5_ARCO
from downscale.rain_resume import atomic_json
from downscale.precipitation import check_pack
from scripts.precipitation import write_pack


def periods(cfg):
    result = []
    year = cfg['annual']['year']
    for month in range(1, 13):
        start = pd.Timestamp(year=year, month=month, day=1)
        end = start + pd.offsets.MonthBegin()
        result.append(('train', start, end))
    for year, months, role in [(cfg['annual']['validation_year'], range(1, 13), 'validation'),
                               (cfg['annual']['test_year'], [1, 4, 7, 10], 'test')]:
        for month in months:
            start = pd.Timestamp(year=year, month=month, day=15)
            result.append((role, start, start + pd.Timedelta(days=1)))
    return result


def hour_ends(start, end):
    return pd.date_range(start + pd.Timedelta(hours=1), end, freq='h')


def atomic_array(path, values):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def aggregate_quarters(values):
    """Instantaneous m/hour rates -> depth in mm over the preceding hour."""
    if len(values) % 4:
        raise ValueError('four quarter-hour rates required per hour')
    return values.reshape(-1, 4, *values.shape[1:]).sum(1, dtype=np.float32) * 250


def tile_indices(source, target, size):
    # Preserve target ordering, including descending native latitude and Greenwich.
    index = pd.Index(source).get_indexer(target, method='nearest', tolerance=1e-5)
    if (index < 0).any():
        raise ValueError('requested grid does not align to PARDIG')
    return [(np.flatnonzero(index // size == k), index[index // size == k])
            for k in np.unique(index // size)]


def targets(times, grid, cache, workers=4):
    tiles = cache / 'tiles'
    tiles.mkdir(exist_ok=True)
    source = open_store(PARDIG_TIMESERIES)
    if source.precipitation.attrs.get('units') != 'm h**-1':
        raise ValueError('unexpected PARDIG rate units')
    quarter = (times.values[:, None] - np.array([45, 30, 15, 0], dtype='timedelta64[m]')).ravel()
    ti = pd.Index(source.time.values).get_indexer(quarter)
    if (ti < 0).any():
        raise ValueError('PARDIG lacks requested quarter-hour targets')
    bucket, prefix = PARDIG_TIMESERIES[5:].split('/', 1)
    store = zarr.storage.ObjectStore(obstore.store.GCSStore(bucket=bucket, prefix=prefix, skip_signature=True))
    rate = zarr.open_group(store=store, mode='r')['precipitation']
    ys = tile_indices(source.latitude.values, grid.lat.values, rate.chunks[1])
    xs = tile_indices(source.longitude.values, grid.lon.values % 360, rate.chunks[2])
    jobs = [(n, y, sy, x, sx) for n, ((y, sy), (x, sx)) in enumerate((a, b) for a in ys for b in xs)]
    def fetch(job):
        n, y, sy, x, sx = job
        path = tiles / f'{n:04d}.npy'
        expected = (len(times), len(y), len(x))
        if path.exists():
            if np.load(path, mmap_mode='r').shape != expected:
                raise ValueError(f'invalid cached tile {path}')
            return
        for attempt in range(5):
            try:
                hourly = aggregate_quarters(rate.oindex[ti, sy, sx])
                if hourly.shape != expected or not np.isfinite(hourly).all() or (hourly < 0).any():
                    raise ValueError(f'invalid target tile {n}')
                atomic_array(path, hourly)
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, job) for job in jobs]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 20 == 0 or done == len(jobs):
                progress = dict(stage='pardig_tiles', completed=done, total=len(jobs), target_hours=len(times),
                                decoded_source_bytes=len(jobs) * int(np.prod(rate.chunks)) * 4)
                atomic_json(cache / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    path = cache / 'targets.npy'
    if not path.exists():
        temporary = cache / 'targets.tmp.npy'
        out = np.lib.format.open_memmap(temporary, mode='w+', dtype='float32', shape=(len(times), len(grid.lat), len(grid.lon)))
        for n, y, sy, x, sx in jobs:
            out[:, y[:, None], x] = np.load(tiles / f'{n:04d}.npy')
        out.flush()
        del out
        temporary.replace(path)
    return np.load(path, mmap_mode='r')


def day_pack(source, times, truth, grid, cfg):
    arrays = []
    hours = cfg['forecast_interval_hours']
    for v in cfg['predictors']:
        stamps = times if v == 'tp' else times[hours - 1::hours]
        field = source[ERA5_NAMES[v]].sel(time=stamps).load()
        if v != 'tp':
            field = field.isel(time=np.repeat(np.arange(len(stamps)), hours)).assign_coords(time=times)
        field = field.interp(lat=grid.lat, lon=grid.lon).astype('float32')
        if v == 'tp':
            field = (field * 1000).clip(min=0)
        arrays.append(field.expand_dims(variable=[v]))
    ds = xr.Dataset({'predictors': xr.concat(arrays, dim='variable').transpose('time', 'variable', 'lat', 'lon'),
                     'target': (('time', 'lat', 'lon'), truth)},
                    coords={'time': times, 'lat': grid.lat, 'lon': grid.lon},
                    attrs={'precipitation_units': 'mm', 'interval': '(time-1h,time]',
                           'target_source': PARDIG_TIMESERIES, 'coarse_source': ERA5_ARCO,
                           'target_resolution_degrees': .1, 'task': 'hourly precipitation downscaling',
                           'satellite_conditioning': 'none'})
    ds.target.attrs = {'units': 'mm', 'interval': '(time-1h,time]'}
    ds = condition_on_intervals(ds, hours)
    check_pack(ds)
    return ds


def prepare(cfg, cache):
    groups = periods(cfg)
    times = pd.DatetimeIndex(np.concatenate([hour_ends(s, e).values for _, s, e in groups]))
    if not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError('train, validation and test years must be chronological and disjoint')
    grid = xr.open_zarr(cfg['static_path']).load()
    fingerprint = hashlib.sha256(times.values.tobytes() + grid.lat.values.tobytes() + grid.lon.values.tobytes()
                                 + json.dumps({'predictors': cfg['predictors'], 'hours': cfg['forecast_interval_hours'],
                                               'source': PARDIG_TIMESERIES, 'era5': ERA5_ARCO, 'schema': 1}, sort_keys=True).encode()).hexdigest()
    manifest = dict(fingerprint=fingerprint, periods=[dict(role=r, start=str(s), end=str(e)) for r,s,e in groups],
                    hours=len(times), grid=[len(grid.lat), len(grid.lon)])
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / 'manifest.json').exists() and json.loads((cache / 'manifest.json').read_text()) != manifest:
        raise ValueError('cache manifest differs; use another cache directory')
    atomic_json(cache / 'manifest.json', manifest)
    truth = targets(times, grid, cache, cfg['annual'].get('download_workers', 4))
    box = cfg['region']
    source = crop(open_store(ERA5_ARCO)[list(ERA5_NAMES.values())], (box[0]-1, box[1]+1, box[2]-1, box[3]+1))
    paths = {'train': [], 'validation': [], 'test': []}
    days = [(role, start, end, day) for role,start,end in groups for day in pd.date_range(start,end,inclusive='left',freq='D')]
    for i, (role, start, end, day) in enumerate(days):
        stamps = hour_ends(day, day + pd.Timedelta(days=1))
        path = cache / 'days' / f'{day:%Y%m%d}.zarr'
        paths[role].append(path)
        if not path.exists():
            temporary = path.with_name(path.name + '.tmp')
            if temporary.exists():
                shutil.rmtree(temporary)
            index = times.get_indexer(stamps)
            if (index < 0).any():
                raise ValueError('day missing from target manifest')
            ds = day_pack(source, stamps, np.asarray(truth[index]), grid, cfg)
            write_pack(ds, path)
        progress = dict(stage='era5_days', completed=i+1, total=len(days), day=str(day), role=role)
        atomic_json(cache / 'progress.json', progress)
        print(json.dumps(progress), flush=True)
    for name, inputs in [('train_validation', paths['train'] + paths['validation']), ('test', paths['test'])]:
        path = cache / f'{name}.zarr'
        if not path.exists():
            temporary = path.with_name(path.name + '.tmp')
            if temporary.exists():
                shutil.rmtree(temporary)
            ds = xr.open_mfdataset([str(p) for p in inputs], engine='zarr', combine='nested', concat_dim='time')
            write_pack(ds.chunk({'time': 24, 'variable': -1, 'lat': -1, 'lon': -1}), path)
            ds.close()
    atomic_json(cache / 'progress.json', dict(stage='preparation_complete', hours=len(times)))


def evaluate_once(command, model, pairs, output, fingerprint):
    """Reuse completed reports; safely restart a partially written evaluation."""
    output = Path(output)
    expected = dict(model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),
                    data_and_config=fingerprint, members=8)
    marker = output / 'complete.json'
    if marker.exists() and json.loads(marker.read_text()) == expected and (output / 'evaluation.json').exists() and (output / 'predictions.zarr').exists():
        return
    for name in ('predictions.zarr', 'predictions.zarr.tmp'):
        path = output / name
        if path.exists():
            shutil.rmtree(path)
    subprocess.run(command + ['evaluate', '--model', str(model), '--pairs', str(pairs), '--out', str(output)], check=True)
    atomic_json(marker, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config_precipitation_year.yaml')
    parser.add_argument('--cache', default='data/precipitation/year_2019')
    parser.add_argument('--out', default='runs/precipitation_year')
    parser.add_argument('--outputs', default='outputs/precipitation_year')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cache, out = Path(args.cache), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f'an annual pipeline is already running in {out}') from None
    atomic_json(out / 'run_config.json', cfg)
    atomic_json(out / 'status.json', dict(stage='preparing'))
    for attempt in range(8):
        try:
            prepare(cfg, cache)
            break
        except (OSError, RuntimeError) as error:
            atomic_json(out / 'status.json', dict(stage='retrying_preparation', attempt=attempt+1, error=str(error)))
            if attempt == 7:
                raise
            time.sleep(30)
    if args.prepare_only:
        return
    command = [sys.executable, '-u', 'scripts/precipitation.py', '--config', args.config]
    atomic_json(out / 'status.json', dict(stage='training'))
    subprocess.run(command + ['train', '--pairs', str(cache / 'train_validation.zarr'), '--out', str(out)], check=True)
    atomic_json(out / 'status.json', dict(stage='evaluating'))
    output = Path(args.outputs)
    fingerprint = json.loads((cache / 'manifest.json').read_text())['fingerprint']
    evaluation_key = fingerprint + json.dumps(cfg, sort_keys=True)
    evaluate_once(command, out / 'model.pt', cache / 'test.zarr', output, evaluation_key)
    previous = output / 'previous_terrain'
    evaluate_once(command, cfg['training']['init_checkpoint'], cache / 'test.zarr', previous, evaluation_key)
    reports = {name: json.loads(path.read_text()) for name, path in
               [('annual', output / 'evaluation.json'), ('previous_terrain', previous / 'evaluation.json')]}
    atomic_json(output / 'comparison.json', reports)
    # A continuous held-out day is easier to inspect than a movie spanning seasons.
    movie_pairs = output / 'movie_pairs.zarr'
    movie_pred = output / 'movie_predictions.zarr'
    for src, dst in [(cache / 'test.zarr', movie_pairs), (output / 'predictions.zarr', movie_pred)]:
        for old in (dst, dst.with_name(dst.name + '.tmp')):
            if old.exists():
                shutil.rmtree(old)
        write_pack(xr.open_zarr(src).isel(time=slice(0,24)).load(), dst)
    subprocess.run([sys.executable, 'scripts/movie_precipitation.py', '--pairs', str(movie_pairs),
                    '--predictions', str(movie_pred), '--out', str(output / 'france_rainfall_year.mp4')], check=True)
    atomic_json(out / 'status.json', dict(stage='complete', evaluation=str(output / 'evaluation.json'), movie=str(output / 'france_rainfall_year.mp4')))


if __name__ == '__main__':
    main()
