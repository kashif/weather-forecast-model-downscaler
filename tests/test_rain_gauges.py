"""Gauge units, missingness and interval alignment are scientific correctness checks."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scripts.fetch_rain_gauges import select_gauge_rows
from scripts.check_rain_gauges import quality_control,aligned_observations,amount_scores
from scripts.check_rain_alerts import probabilities


def test_gauge_units_quality_and_missing_are_preserved():
    times=pd.date_range('2025-01-01',periods=5,freq='h')
    frame=pd.DataFrame(dict(NUM_POSTE=['1']*5,NOM_USUEL=['A']*5,LAT=[45.]*5,LON=[2.]*5,ALTI=[100]*5,
                            AAAAMMJJHH=times.strftime('%Y%m%d%H'),RR1=[12.3,0,np.nan,5,-1],QRR1=[1,0,1,2,1]))
    result=select_gauge_rows(frame,times,[0,1])
    np.testing.assert_allclose(result.rain_mm,[12.3,0]);assert result.station.tolist()==['00000001']*2


def test_conflicts_completeness_and_missing_alignment():
    times=pd.date_range('2025-01-01T01',periods=4,freq='h')
    rows=pd.DataFrame(dict(station=['a']*5+['b'],time=list(times)+[times[0],times[0]],rain_mm=[1,2,3,4,9,2],
                           lat=[45.]*6,lon=[2.]*6,elevation_m=[100.]*6,name=['A']*6))
    rows,stations,qc=quality_control(rows,times,[40,53,-7,11],.75)
    assert stations.index.tolist()==['a'];assert qc['conflicting_rows']==2
    values=aligned_observations(rows,stations,times)
    assert np.isnan(values[0,0]);np.testing.assert_equal(values[1:,0],[2,3,4])
    shifted=aligned_observations(rows,stations,times,1)
    np.testing.assert_equal(shifted[:3,0],[2,3,4]);assert np.isnan(shifted[-1,0])


def test_no_fake_target_needed_for_probability_prediction():
    data=SimpleNamespace(rain=np.array([[0,8],[10,12]]))
    np.testing.assert_equal(probabilities(data,None,[5,10]),[[.5,0],[1,.5]])


def test_accumulations_require_complete_windows():
    obs=np.ones((24,2));pred=obs*2;obs[0,0]=np.nan
    result=amount_scores(pred,obs,6)
    assert result['windows']==7;assert result['bias_mm']==6
    assert amount_scores(pred,obs,24)['windows']==1


def test_bilinear_probability_sampling_does_not_threshold_mean_rain():
    import xarray as xr
    from downscale.station_interp import GridInterpolator
    probability=xr.DataArray([[0.,1.],[0.,1.]],dims=['lat','lon'])
    sampler=GridInterpolator.build([0,1],[0,1],[.5],[.5],'bilinear')
    assert sampler.apply(probability).item()==.5
    # The equal-weight mixture retains half the event probability.
    rain=xr.DataArray([[0.,8.],[0.,8.]],dims=['lat','lon'])
    assert not (sampler.apply(rain).item()>5)


def test_no_complete_daily_observations_produces_no_score():
    obs=np.ones((24,1));obs[12]=np.nan
    result=amount_scores(np.ones_like(obs),obs,24)
    assert result==dict(windows=0,rmse_mm=None,bias_mm=None,mae_mm=None)
