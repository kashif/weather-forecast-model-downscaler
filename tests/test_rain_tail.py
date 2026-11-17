"""POT calibration likelihood and analytic scores match their distributions."""
import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import expit
from downscale.rain_tail import (TailData,TailHead,fit_tail,TailCalibration,gpd_survival,
    empirical_crps,spliced_crps,spliced_twcrps,spliced_survival,spliced_moments)


@pytest.mark.parametrize('xi',[0.,.15,.4])
def test_gpd_crps_matches_numerical_cdf_integration(xi):
    rain=np.array([[0.,.2,1.,4.]])
    u=2.;p=np.array([.25]);scale=np.array([1.3])
    for target in [0.,.5,2.,7.]:
        def integrand(z):
            cdf=1-spliced_survival(rain,z,u,p,scale,xi)[0]
            return (cdf-float(z>=target))**2
        edges=sorted({0.,.2,1.,u,target,50.})
        numerical=sum(quad(integrand,a,b,epsabs=1e-10)[0] for a,b in zip(edges[:-1],edges[1:]))+quad(integrand,50,np.inf,epsabs=1e-10)[0]
        actual=spliced_crps(rain,np.array([target]),u,p,scale,xi)[0]
        assert actual==pytest.approx(numerical,rel=2e-6,abs=2e-8)


def test_splice_occurrence_and_body_order_are_correct():
    # Raw input deliberately unsorted; the body must not use sorted masks on it.
    rain=np.array([[8.,0.,1.,4.],[8.,9.,10.,11.]])
    p=np.array([.2,.3]);s=np.ones(2);u=2.;xi=.1
    np.testing.assert_allclose(spliced_survival(rain,.5,u,p,s,xi),[.6,1.])
    np.testing.assert_allclose(spliced_survival(rain,u,u,p,s,xi),p)
    np.testing.assert_allclose(spliced_crps(rain,np.array([0.,3.]),u,np.zeros(2),s,xi),[.25,1.])
    assert np.all(spliced_crps(rain,np.array([0.,3.]),u,p,s,xi)>=0)


def test_tail_score_is_crps_of_censored_distribution_and_moments_are_finite():
    rain=np.array([[0.,1.,2.,3.]])
    y=np.array([9.]);p=np.array([.3]);s=np.array([2.]);xi=.2;u=3.;c=5.
    actual=spliced_twcrps(y,c,u,p,s,xi)
    pc=p*gpd_survival(c-u,s,xi)
    expected=spliced_crps(np.array([[c]]),np.maximum(y,c),c,pc,s+xi*(c-u),xi)
    np.testing.assert_allclose(actual,expected)
    mean,var=spliced_moments(rain,u,np.ones(1),s,xi)
    assert mean[0]==pytest.approx(u+2/.8)
    assert var[0]==pytest.approx(4/(.8**2*.6))
    with pytest.raises(ValueError):spliced_twcrps(y,1.,u,p,s,xi)


def test_empirical_crps_matches_pairwise_definition():
    rng=np.random.default_rng(9);r=rng.exponential(size=(20,8));y=rng.exponential(size=20)
    direct=np.abs(r-y[:,None]).mean(1)-.5*np.abs(r[:,:,None]-r[:,None,:]).mean((1,2))
    np.testing.assert_allclose(empirical_crps(r,y),direct,atol=1e-14)


def test_fit_learns_conditional_exceedances_and_checkpoint_roundtrips(tmp_path):
    torch.set_num_threads(1)
    rng=np.random.default_rng(11);n=4000;x=rng.normal(size=(n,2)).astype('float32')
    probability=expit(-2+1.2*x[:,0]);event=rng.random(n)<probability
    scale=np.logaddexp(0,.2+.3*x[:,0])+.2;xi=.15
    excess=scale/xi*((1-rng.random(n))**(-xi)-1)
    y=np.where(event,2+excess,0).astype('float32')
    raw=np.zeros((n,8),dtype='float32')
    data=TailData(x,raw,y,np.arange(n)%4,['a','b','c','d'],['first','second'])
    cal=fit_tail(data,2.,dict(max_iterations=70,max_negative_fit_points=1000,regularization=.0001))
    p,s,shape=cal.parameters(x,raw)
    assert np.corrcoef(p,probability)[0,1]>.95
    assert np.mean((p-event)**2)<np.mean((event.mean()-event)**2)
    assert np.all(s>0) and 0<shape<.45
    assert cal.metadata['exceedance_days']==4
    path=tmp_path/'tail.pt';cal.save(path);restored=TailCalibration.load(path)
    p2,s2,shape2=restored.parameters(x,raw)
    np.testing.assert_array_equal(p,p2);np.testing.assert_array_equal(s,s2);assert shape==shape2


def test_shape_constraints_keep_sample_variance_defined():
    with pytest.raises(ValueError):TailHead(2,(0.,.6))
    head=TailHead(2)
    logits,scale,xi=head(torch.zeros(3,2,dtype=torch.float64))
    loss=-torch.distributions.GeneralizedPareto(torch.zeros(3),scale.float(),xi.float()).log_prob(torch.ones(3)).mean()
    loss.backward()
    assert torch.isfinite(head.shape_raw.grad)


def test_selection_guard_and_disjoint_dates():
    from scripts.calibrate_rain_tail import choose_candidate,validate_splits
    candidates=[dict(selection_crps_mm=1.06,selection_tail_objective=.1),
                dict(selection_crps_mm=1.02,selection_tail_objective=.2)]
    assert choose_candidate(candidates,1.,.3,1.05)==1
    assert choose_candidate(candidates,1.,.15,1.05) is None
    cfg={'splits':{'fit':['2021-01-15'],'selection':['2023-01-15'],'test':['2024-01-15']}}
    cp={'data_attrs':{'training_start':'2019-01-01','training_end':'2020-01-01',
                      'validation_start':'2020-01-15','validation_end':'2020-12-16'}}
    validate_splits(cfg,cp)
    cfg['splits']['test']=['2021-01-15']
    with pytest.raises(ValueError):validate_splits(cfg,cp)
    cfg['splits']['test']=['2024-01-15'];cfg['splits']['fit']=['2019-02-01']
    with pytest.raises(ValueError,match='frozen-model'):validate_splits(cfg,cp)


def test_features_use_forecasts_and_static_terrain_without_target_values():
    import pandas as pd
    import xarray as xr
    from scripts.calibrate_rain_tail import forecast_features,FEATURE_NAMES
    n,h,w=6,3,4
    variables=['tp','t2m','d2m','u10m','v10m','msl','phase']
    values=np.broadcast_to(np.array([1,280,279,3,4,100000,.5],dtype='float32')[None,:,None,None],(n,7,h,w)).copy()
    ds=xr.Dataset({'predictors':(('time','variable','lat','lon'),values),
                   'target':(('time','lat','lon'),np.ones((n,h,w),dtype='float32'))},
                  coords={'time':pd.date_range('2021-01-15T01',periods=n,freq='h'),'variable':variables,
                          'lat':np.arange(h,dtype=float),'lon':np.arange(w,dtype=float)},attrs={'precipitation_units':'mm'})
    pred=xr.Dataset({'precipitation':(('time','sample','lat','lon'),np.ones((n,4,h,w),dtype='float32'))},
                    coords={'time':ds.time,'lat':ds.lat,'lon':ds.lon,'sample':np.arange(4)})
    pred.precipitation.attrs['units']='mm'
    static=np.ones((8,h,w),dtype='float32')
    a=forecast_features(ds,pred,static)
    ds['target'][:]=np.nan
    b=forecast_features(ds,pred,static)
    assert a.shape==(n*h*w,len(FEATURE_NAMES))
    np.testing.assert_array_equal(a,b)
    assert np.all(np.isfinite(a))


def test_exponential_special_case_fits_and_reloads(tmp_path):
    rng=np.random.default_rng(2);n=500
    x=rng.normal(size=(n,1)).astype('float32');y=np.zeros(n,dtype='float32')
    y[::3]=5+rng.exponential(size=len(y[::3]))
    rain=np.zeros((n,4),dtype='float32')
    data=TailData(x,rain,y,np.zeros(n,dtype=int),['day'],['feature'])
    model=fit_tail(data,5.,dict(fixed_shape=0.,max_iterations=25))
    assert model.parameters(x,rain)[2]==0.
    model.save(tmp_path/'exp.pt')
    restored=TailCalibration.load(tmp_path/'exp.pt')
    assert restored.parameters(x,rain)[2]==0.
    assert np.isfinite(restored.parameters(x,rain)[1]).all()
