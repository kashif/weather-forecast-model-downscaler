"""Annual reads preserve hour-end semantics and resumes preserve optimizer/noise state."""
import copy
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from scripts.precipitation_year import aggregate_quarters, tile_indices, periods, hour_ends, day_pack
from downscale.precip_data import ERA5_NAMES, build_pairs, condition_on_intervals


def test_annual_split_and_native_tile_indices():
    cfg = {'annual': {'year': 2019, 'validation_year': 2020, 'test_year': 2022}}
    groups = periods(cfg)
    assert [sum(len(hour_ends(s,e)) for r,s,e in groups if r == role)
            for role in ['train','validation','test']] == [8760,288,96]
    source = np.arange(10)[::-1]
    groups = tile_indices(source, np.array([1,2,6,8]), 5)
    restored = np.empty(4)
    for target, native in groups:
        assert len(np.unique(native // 5)) == 1
        restored[target] = source[native]
    np.testing.assert_array_equal(restored, [1,2,6,8])
    values = np.arange(1,9, dtype='float32')[:,None,None] / 1000
    np.testing.assert_allclose(aggregate_quarters(values).ravel(), [2.5,6.5])
    with pytest.raises(ValueError):
        aggregate_quarters(values[:3])


def test_daily_preparation_matches_existing_pair_semantics(monkeypatch):
    import downscale.precip_data as data
    times = pd.date_range('2019-01-01T01', periods=24, freq='h')
    lat,lon = np.arange(3.), np.arange(4.)
    shape = (24,3,4)
    source = xr.Dataset({name: (('time','lat','lon'), np.broadcast_to(np.arange(24)[:,None,None] + i + 1,shape).astype('float32'))
                         for i,name in enumerate(ERA5_NAMES.values())},coords=dict(time=times,lat=lat,lon=lon))
    source[ERA5_NAMES['tp']] /= 1000
    def fields(stamps, variables, backend, box):
        return xr.concat([source[ERA5_NAMES[v]].sel(time=stamps).expand_dims(variable=[v]) for v in variables],dim='variable').transpose('time','variable','lat','lon')
    monkeypatch.setattr(data,'era5_fields',fields)
    truth = xr.DataArray(np.ones(shape,dtype='float32') / 1000,dims=('time','lat','lon'),coords=dict(time=times,lat=lat,lon=lon))
    cfg = dict(predictors=list(ERA5_NAMES),forecast_interval_hours=6)
    expected = condition_on_intervals(build_pairs(times, target=truth, forcing_hours=6),6)
    actual = day_pack(source,times,truth.values * 1000,source,cfg)
    xr.testing.assert_allclose(actual.predictors,expected.predictors)
    xr.testing.assert_allclose(actual.target,expected.target)


def test_mid_epoch_resume_reproduces_uninterrupted_training(tmp_path, monkeypatch):
    from scripts.precipitation import train
    import downscale.rain_resume as resume
    n,h,w=36,5,7
    rain=np.ones((n,h,w),dtype='float32')
    phase=np.broadcast_to(np.tile(np.arange(1,7)/6,6)[:,None,None],rain.shape).astype('float32')
    ds=xr.Dataset({'predictors':(('time','variable','lat','lon'),np.stack([rain,phase],1)),
                   'target':(('time','lat','lon'),rain*.8)},coords={'time':pd.date_range('2020-01-01T01',periods=n,freq='h'),
                   'variable':['tp','phase'],'lat':np.arange(h),'lon':np.arange(w)},attrs={'forecast_interval_hours':6,'precipitation_units':'mm','interval':'(time-1h,time]'})
    cfg={'validation_start':'2020-01-02T01','model':{'base':4,'depth':2,'members':2,'architecture':'joint','sequence_hours':12},
         'training':{'seed':0,'epochs':2,'batch_windows':1,'sequence_windows':2,'learning_rate':.0001,'temporal_weight':.1,
                     'selection':'joint','resume':True,'checkpoint_every_steps':1}}
    train(ds,copy.deepcopy(cfg),tmp_path/'full','cpu')
    original=resume.save_resume
    def interrupted(*args,**kwargs):
        original(*args,**kwargs)
        raise RuntimeError('simulated power interruption')
    monkeypatch.setattr(resume,'save_resume',interrupted)
    with pytest.raises(RuntimeError, match='power interruption'):
        train(ds,copy.deepcopy(cfg),tmp_path/'resumed','cpu')
    monkeypatch.setattr(resume,'save_resume',original)
    train(ds,copy.deepcopy(cfg),tmp_path/'resumed','cpu')
    full=torch.load(tmp_path/'full'/'resume.pt',weights_only=True)
    restored=torch.load(tmp_path/'resumed'/'resume.pt',weights_only=True)
    assert full['history'] == restored['history']
    for key,value in full['state_dict'].items():
        torch.testing.assert_close(value,restored['state_dict'][key],rtol=0,atol=0)
    changed=copy.deepcopy(cfg); changed['training']['learning_rate']=.01
    with pytest.raises(ValueError,match='resume data/config'):
        train(ds,changed,tmp_path/'resumed','cpu')
