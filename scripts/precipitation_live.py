"""Download one real global run, downscale hourly rainfall, and animate France.

Uses byte-range ECMWF open-data reads. Samples describe downscaler uncertainty
conditional on a single deterministic global forecast, not an NWP ensemble.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import xarray as xr
import torch
from downscale.open_data import _get, file_url, decode
from downscale.precip_data import crop, forecast_pack
from downscale.precipitation import predict, check_pack
from downscale.hourly import default_device
from scripts.precipitation import write_pack

PARAMS = {'tp': 'tp', 't2m': '2t', 'd2m': '2d', 'u10m': '10u', 'v10m': '10v', 'msl': 'msl'}
UNITS = {'tp': ('m', 'kg m**-2'), 't2m': 'K', 'd2m': 'K', 'u10m': 'm s**-1', 'v10m': 'm s**-1', 'msl': 'Pa'}


def validate_message(message, variable, init, lead):
    import eccodes
    h = eccodes.codes_new_from_message(message)
    try:
        units = eccodes.codes_get(h, 'units')
        allowed = UNITS[variable] if isinstance(UNITS[variable], tuple) else (UNITS[variable],)
        if units not in allowed:
            raise ValueError(f'{variable}: unexpected units {units}')
        expected = {'shortName': PARAMS[variable],
                    'dataDate': int(pd.Timestamp(init).strftime('%Y%m%d')),
                    'dataTime': pd.Timestamp(init).hour * 100, 'endStep': lead}
        if variable == 'tp':
            expected.update(stepType='accum', startStep=0)
        for key, value in expected.items():
            actual = eccodes.codes_get(h, key)
            if actual != value:
                raise ValueError(f'{variable} +{lead}: {key}={actual}, expected {value}')
        return units
    finally:
        eccodes.codes_release(h)


def fetch_fields(init, hours, cache, model='aifs-single'):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    def step(lead):
        url = file_url(init, lead, 'fc', model=model)
        index_path = cache / f'{lead:03d}.index'
        if not index_path.exists():
            index_path.write_bytes(_get(url + '.index'))
        records = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
        fields = []
        for variable, param in PARAMS.items():
            matches = [r for r in records if r.get('param') == param and r.get('levtype') == 'sfc']
            if len(matches) != 1:
                raise ValueError(f'{url}: expected one {param} surface field')
            record = matches[0]
            path = cache / f'{lead:03d}_{variable}.grib2'
            if not path.exists():
                message = _get(url + '.grib2', (record['_offset'], record['_length']))
                if len(message) != record['_length']:
                    raise ValueError('server returned an unexpected byte range')
                validate_message(message, variable, init, lead)
                temporary = path.with_suffix('.tmp')
                temporary.write_bytes(message)
                temporary.replace(path)
            message = path.read_bytes()
            units = validate_message(message, variable, init, lead)
            _, lat, lon, values = decode(message)
            field = xr.DataArray(values, dims=('latitude', 'longitude'), coords={'latitude': lat, 'longitude': lon})
            field = crop(field, (39,54,-8,12)).astype('float32')
            if variable == 'tp' and units == 'kg m**-2':
                field = field / 1000  # kg/m² liquid-water equivalent is mm; schema requires m
            fields.append(field.expand_dims(variable=[variable]))
        result = xr.concat(fields, dim='variable').expand_dims(lead_time=[np.timedelta64(lead,'h')])
        print(f'fetched {model} {init} +{lead}h', flush=True)
        return lead, result
    arrays = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(step, lead) for lead in range(6,hours+1,6)]
        for future in as_completed(futures):
            lead, array = future.result()
            arrays[lead] = array
    fields = xr.concat([arrays[k] for k in sorted(arrays)], dim='lead_time').assign_coords(time=np.datetime64(init, 'ns'))
    fields.name = 'fields'
    fields.attrs = {'source': f'ECMWF {model} open data', 'precipitation_units': 'm',
                    'precipitation_accumulation': 'cumulative since initialization',
                    'licence': 'CC-BY-4.0; ECMWF', 'source_base': 'https://storage.googleapis.com/ecmwf-open-data'}
    return fields.transpose('lead_time','variable','lat','lon')


def interval_fields(fields, packing_error_m=None):
    result = fields.copy(deep=True)
    total = fields.sel(variable='tp').values
    depth = np.diff(total, axis=0, prepend=np.zeros_like(total[:1]))
    if not np.isfinite(depth).all():
        raise ValueError('cumulative rainfall resets or missing values')
    if packing_error_m is None:
        tolerance = np.full((len(total),1,1),1e-6)
    else:
        error = np.asarray(packing_error_m)
        if error.shape != (len(total),) or not np.isfinite(error).all() or (error<0).any():
            raise ValueError('invalid GRIB packing error')
        tolerance = (error + np.r_[0,error[:-1]])[:,None,None] + 1e-8
    if (depth < -tolerance).any():
        raise ValueError('cumulative rainfall resets or missing values')
    # Quantization of independent cumulative messages can yield small decreases.
    # A monotone envelope avoids adding rainfall at every down/up quantization step.
    monotone = np.maximum.accumulate(total,axis=0)
    if packing_error_m is not None:
        bound = (np.maximum.accumulate(error) + error)[:,None,None] + 1e-8
        if ((monotone-total)>bound).any():
            raise ValueError('cumulative correction exceeds GRIB quantization uncertainty')
    result.loc[dict(variable='tp')] = np.diff(monotone,axis=0,prepend=np.zeros_like(total[:1]))
    result.attrs['max_cumulative_quantization_correction_mm'] = float((monotone-total).max()*1000)
    result.attrs['precipitation_accumulation'] = 'depth per six-hour interval'
    return result


def conditional_ensemble(checkpoint, forcing, members=8, device=None):
    """Stable groups of eight: expanding an ensemble preserves earlier samples."""
    if members < 2:
        raise ValueError('at least two samples required for sample variance')
    blocks=[]
    for start in range(0,members,8):
        size=min(8,members-start)
        block=predict(checkpoint,forcing,device=device or default_device(),members=8,noise_seed=2+start//8)
        block=block.isel(sample=slice(0,size))
        blocks.append(block)
        print(f'generated {start+size}/{members} conditional samples',flush=True)
    result=xr.concat(blocks,dim='sample').assign_coords(sample=np.arange(members))
    result.attrs.update(noise_sampling='groups of up to 8; group seeds 2,3,...',conditional_samples=members)
    return result


def ensemble_statistics(rain):
    """Variance of hourly depth in mm², distinct from variance of its sample mean."""
    if rain.shape[1]<2:
        raise ValueError('at least two samples required for sample variance')
    return rain.mean(1),rain.var(1,ddof=1),(rain>=1).mean(1)


def make_forecast_movie(forcing, prediction, path, init, model, first_hour=None, fps=3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.dates as mdates
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    check_pack(forcing, target=False)
    if first_hour:
        forcing = forcing.sel(time=slice(first_hour,None))
        prediction = prediction.sel(time=forcing.time)
    for name in ['time','lat','lon']:
        if not np.array_equal(forcing[name], prediction[name]):
            raise ValueError(f'forecast/movie {name} mismatch')
    times = pd.DatetimeIndex(forcing.time.values)
    if not len(times) or not np.all(np.diff(times.values) == np.timedelta64(1,'h')):
        raise ValueError('movie needs consecutive hourly predictions')
    rain = prediction.precipitation.values
    coarse = forcing.predictors.sel(variable='tp').values
    mean,variance,probability=ensemble_statistics(rain)
    fields = [coarse,mean,variance,probability]
    titles = ['Global forcing\nSix-hour rainfall / 6',f'Downscaled mean\n{rain.shape[1]} terrain-conditioned samples',
              'Conditional sample variance\nSpread of hourly rainfall (mm²)', 'Probability of ≥1 mm in this hour\nConditional on this global run']
    palette = ['#ffffff','#c9e9f6','#80c6e3','#3596c5','#2268a7','#33428c','#7950a0','#c65887','#e6874a']
    cmap = ListedColormap(palette)
    norm = BoundaryNorm([0,.1,.5,1,2,5,10,20,40,100],cmap.N)
    fig = plt.figure(figsize=(15,6.8),facecolor='#f8fafc')
    grid = fig.add_gridspec(2,4,height_ratios=[3,1.1],left=.055,right=.97,bottom=.15,top=.79,wspace=.12,hspace=.28)
    meshes=[]
    for i,(field,title) in enumerate(zip(fields,titles)):
        ax=fig.add_subplot(grid[0,i],projection=ccrs.PlateCarree())
        ax.set_extent([float(forcing.lon.min()),float(forcing.lon.max()),float(forcing.lat.min()),float(forcing.lat.max())])
        mesh=ax.pcolormesh(forcing.lon,forcing.lat,field[0],shading='auto',transform=ccrs.PlateCarree(),
                           **({'cmap':cmap,'norm':norm} if i<2 else
                              {'cmap':'YlOrRd','vmin':0,'vmax':10} if i==2 else
                              {'cmap':'YlGnBu','vmin':0,'vmax':1}))
        ax.coastlines('50m',linewidth=.45,color='#56616e')
        ax.add_feature(cfeature.BORDERS.with_scale('50m'),linewidth=.35,edgecolor='#7a8490')
        ax.set_title(title,fontsize=9,loc='left',pad=9)
        meshes.append(mesh)
    for mesh,position,label,ticks in [
            (meshes[0],[.055,.385,.43,.012],'Hourly rainfall (mm)',[.1,1,5,10,20,40]),
            (meshes[2],[.535,.385,.19,.012],'Sample variance (mm²)',[0,2,5,10]),
            (meshes[3],[.78,.385,.19,.012],'Probability',[0,.5,1])]:
        bar=fig.add_axes(position)
        cb=fig.colorbar(mesh,cax=bar,orientation='horizontal',ticks=ticks,extend='max' if mesh is meshes[2] else 'neither')
        cb.set_label(label,fontsize=8,labelpad=1)
        bar.tick_params(labelsize=7,pad=1)
    ax=fig.add_subplot(grid[1,:]);regional=rain.mean((-2,-1))
    ax.plot(times,coarse.mean((-2,-1)),color='#82909f',label='Global six-hour mean')
    ax.plot(times,regional.mean(1),color='#176dba',label='Downscaled mean')
    ax.fill_between(times,np.quantile(regional,.05,axis=1),np.quantile(regional,.95,axis=1),color='#176dba',alpha=.18,label='5–95% conditional sample range')
    ax.set_ylabel('Domain mean\n(mm/hour)',fontsize=9);ax.grid(alpha=.15)
    ax.legend(ncol=3,fontsize=8,loc='upper left');ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
    cursor=ax.axvline(times[0],color='#d37a25')
    fig.text(.055,.95,'Experimental hourly rainfall forecast over France',fontsize=19,weight='bold',color='#172b40')
    fig.text(.055,.90,f'Global forecast: ECMWF {model.upper()} • initialized {pd.Timestamp(init):%d %b %Y %H:%M} UTC',fontsize=11,color='#485b6d')
    stamp=fig.text(.055,.85,'',fontsize=11,color='#485b6d')
    fig.text(.055,.065,'0.1° (~8–11 km) • Epoch-5 full-year terrain checkpoint • Samples share one deterministic global forecast',fontsize=9,color='#5b6d7f')
    fig.text(.055,.03,'Experimental transfer from ERA5 training to AIFS forcing; operational forecast skill has not been independently validated. Data: ECMWF, CC-BY-4.0.',fontsize=8,color='#5b6d7f')
    def update(i):
        for mesh,field in zip(meshes,fields):mesh.set_array(field[i].ravel())
        lead=(times[i]-pd.Timestamp(init))/pd.Timedelta(hours=1)
        stamp.set_text(f'Hour ending {times[i]:%a %d %b %Y %H:%M} UTC  |  lead +{lead:.0f}h  |  {i+1}/{len(times)}')
        cursor.set_xdata([times[i],times[i]])
        return meshes+[cursor,stamp]
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    update(0);fig.savefig(path.with_suffix('.png'),dpi=130)
    movie=animation.FuncAnimation(fig,update,frames=len(times),interval=1000/fps,blit=False)
    try:
        movie.save(path,writer=animation.FFMpegWriter(fps=fps,codec='libx264',bitrate=2800,extra_args=['-pix_fmt','yuv420p','-movflags','+faststart']),dpi=110)
    finally:plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init',required=True)
    parser.add_argument('--hours',type=int,default=96)
    parser.add_argument('--model',default='aifs-single',choices=['aifs-single','ifs'])
    parser.add_argument('--checkpoint',default='runs/precipitation_year/model.pt')
    parser.add_argument('--out',required=True)
    parser.add_argument('--members',type=int,default=8,help='conditional noise samples; stable groups of eight')
    parser.add_argument('--reuse-inputs',help='reuse global_fields.zarr and grib/ from another output directory')
    parser.add_argument('--first-hour',help='optional first displayed hour end in UTC')
    args=parser.parse_args()
    if args.members<2:parser.error('members must be at least two')
    if args.hours<6 or args.hours%6:parser.error('hours must be a positive multiple of six')
    init=pd.Timestamp(args.init)
    if init.tz is not None or init != init.floor('6h'):
        parser.error('initialization must be a naive UTC timestamp on a six-hour boundary')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    manifest=dict(initialization=str(init),hours=args.hours,model=args.model,
                  checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),samples=args.members,noise_seed=2)
    manifest_path=out/'inputs.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('output cache belongs to another forecast or checkpoint; choose another --out')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    inputs=Path(args.reuse_inputs) if args.reuse_inputs else out
    fields_path=inputs/'global_fields.zarr'
    if not fields_path.exists():
        fields=fetch_fields(args.init,args.hours,inputs/'grib',args.model)
        write_pack(fields.to_dataset(),fields_path)
    else:fields=xr.open_zarr(fields_path).fields.load()
    if fields.attrs.get('source')!=f'ECMWF {args.model} open data':
        raise ValueError('cached forcing comes from another global model')
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if fields.time.values != np.datetime64(args.init,'ns') or fields.sizes['lead_time'] != args.hours//6:
        raise ValueError('existing forecast cache differs from requested run/horizon')
    import eccodes
    packing_error=[]
    for lead in fields.lead_time.values / np.timedelta64(1,'h'):
        h=eccodes.codes_new_from_message((inputs/'grib'/f'{int(lead):03d}_tp.grib2').read_bytes())
        try:
            error=float(eccodes.codes_get(h,'packingError'))
            if eccodes.codes_get(h,'units')=='kg m**-2': error/=1000
            packing_error.append(error)
        finally: eccodes.codes_release(h)
    intervals=interval_fields(fields,packing_error)
    forcing=forecast_pack(intervals,cp['lat'].numpy(),cp['lon'].numpy(),hours=6,accumulation='interval')
    forcing.attrs.update(global_model=args.model,forecast_init=args.init,forecast_source=fields.attrs['source'],
                         max_cumulative_quantization_correction_mm=intervals.attrs['max_cumulative_quantization_correction_mm'])
    if not (out/'forcing.zarr').exists():write_pack(forcing,out/'forcing.zarr')
    if not (out/'predictions.zarr').exists():
        prediction=conditional_ensemble(args.checkpoint,forcing,members=args.members)
        prediction.attrs.update(global_model=args.model,forecast_init=args.init,uncertainty='conditional on one deterministic global forecast')
        write_pack(prediction,out/'predictions.zarr')
    else:prediction=xr.open_zarr(out/'predictions.zarr').load()
    if not np.isfinite(prediction.precipitation.values).all() or (prediction.precipitation.values<0).any():
        raise ValueError('nonfinite or negative predicted rainfall')
    movie=out/'france_forecast.mp4'
    make_forecast_movie(forcing,prediction,movie,args.init,args.model,args.first_hour)
    report=dict(initialization=args.init,global_model=args.model,horizon_hours=args.hours,first_displayed_hour=args.first_hour,
                checkpoint=args.checkpoint,checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                terrain=cp['config']['terrain'],conditional_samples=args.members,grid_degrees=.1,
                max_cumulative_quantization_correction_mm=intervals.attrs['max_cumulative_quantization_correction_mm'],
                forecast_max_sample_mm=float(prediction.precipitation.max()),movie=str(movie),
                caveat='ERA5-trained downscaler driven by operational AIFS; skill of this transfer not independently verified')
    (out/'forecast.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
