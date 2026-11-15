"""Operational forecast conversion keeps accumulated rain in physical units."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scripts.precipitation_live import interval_fields, validate_message
from downscale.precip_data import forecast_pack


def test_cumulative_forecast_depths_become_hourly_mm_with_endpoint_states():
    values=np.ones((2,2,3,4),dtype='float32') * 280
    values[:,0]=np.array([.006,.018])[:,None,None]
    fields=xr.DataArray(values,dims=('lead_time','variable','lat','lon'),coords={
        'lead_time':np.array([6,12],dtype='timedelta64[h]'),'variable':['tp','t2m'],
        'lat':np.arange(3.),'lon':np.arange(4.),'time':np.datetime64('2026-11-15T06')})
    result=forecast_pack(interval_fields(fields),fields.lat,fields.lon,accumulation='interval')
    np.testing.assert_allclose(result.predictors.sel(variable='tp').values[:,0,0],[1]*6+[2]*6,rtol=1e-6)
    np.testing.assert_allclose(result.predictors.sel(variable='t2m'),280)
    assert pd.Timestamp(result.time[0].values)==pd.Timestamp('2026-11-15T07')
    assert pd.Timestamp(result.time[-1].values)==pd.Timestamp('2026-11-15T18')
    broken=fields.copy();broken.loc[dict(variable='tp',lead_time=fields.lead_time[-1])]=0
    with pytest.raises(ValueError,match='resets'):
        interval_fields(broken)


def test_grib_validation_rejects_wrong_run_or_interval(monkeypatch):
    import eccodes
    metadata=dict(shortName='tp',units='kg m**-2',dataDate=20261115,dataTime=600,
                  endStep=12,startStep=0,stepType='accum')
    monkeypatch.setattr(eccodes,'codes_new_from_message',lambda message:1)
    monkeypatch.setattr(eccodes,'codes_get',lambda handle,key:metadata[key])
    monkeypatch.setattr(eccodes,'codes_release',lambda handle:None)
    assert validate_message(b'grib','tp','2026-11-15T06',12)=='kg m**-2'
    metadata['startStep']=6
    with pytest.raises(ValueError,match='startStep'):
        validate_message(b'grib','tp','2026-11-15T06',12)
    metadata['startStep']=0
    with pytest.raises(ValueError,match='dataTime'):
        validate_message(b'grib','tp','2026-11-15T00',12)


def test_quantized_accumulation_uses_monotone_envelope_without_extra_water():
    fields=xr.DataArray(np.array([1.,.95,2.],dtype='float32')[:,None,None,None]/1000,
                       dims=('lead_time','variable','lat','lon'),coords={'variable':['tp']})
    result=interval_fields(fields,np.array([.00005]*3))
    np.testing.assert_allclose(result.sel(variable='tp').values.ravel()*1000,[1,0,1],rtol=1e-6)
    assert result.attrs['max_cumulative_quantization_correction_mm']==pytest.approx(.05,abs=1e-6)
    with pytest.raises(ValueError,match='resets'):
        interval_fields(fields,np.array([.000001]*3))


def test_ensemble_growth_preserves_first_eight_and_reports_sample_variance(monkeypatch):
    import scripts.precipitation_live as live
    calls=[]
    def fake_predict(checkpoint,forcing,device,members,noise_seed):
        calls.append((members,noise_seed))
        values=np.arange(members,dtype='float32')+noise_seed*10
        return xr.Dataset({'precipitation':(('time','sample','lat','lon'),values[None,:,None,None])},coords={'sample':np.arange(members)})
    monkeypatch.setattr(live,'predict',fake_predict)
    a=live.conditional_ensemble('cp',None,8,device='cpu')
    b=live.conditional_ensemble('cp',None,32,device='cpu')
    xr.testing.assert_equal(a.precipitation,b.precipitation.isel(sample=slice(0,8)))
    assert calls[-4:]==[(8,2),(8,3),(8,4),(8,5)]
    partial=live.conditional_ensemble('cp',None,10,device='cpu')
    xr.testing.assert_equal(partial.precipitation,b.precipitation.isel(sample=slice(0,10)))
    mean,variance,prob=live.ensemble_statistics(b.precipitation.values)
    np.testing.assert_allclose(mean,b.precipitation.values.mean(1))
    np.testing.assert_allclose(variance,b.precipitation.values.var(1,ddof=1))
    assert np.all(prob==1)
    with pytest.raises(ValueError,match='two samples'):
        live.ensemble_statistics(np.ones((1,1,2,2)))
