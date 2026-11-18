"""Categorical intensity probabilities preserve strict thresholds and population weights."""
import numpy as np
import pytest
import torch
from downscale.rain_categorical import CategoricalRain,BOUNDS,labels,design
from scripts.train_rain_categorical import sampling_indices,predict_probabilities,selection_score
from scripts.check_rain_alerts import probabilities
from scripts.check_rain_gauges import paired_brier_interval
from types import SimpleNamespace


def test_bins_include_dry_point_and_strict_threshold_boundaries():
    values=np.array([0,.01,.1,5,5.01,10,10.01,20,20.01,200],dtype='float32')
    np.testing.assert_equal(labels(values),[0,1,1,5,6,6,7,7,8,9])
    for t in [5,10,20]:
        np.testing.assert_equal(labels(values)>np.flatnonzero(BOUNDS==t)[0],values>t)
    with pytest.raises(ValueError):labels([-1])


def test_population_weights_restore_nonheavy_mass_without_discarding_extremes():
    y=np.r_[np.zeros(100),[6,11,21]]
    index,weights=sampling_indices(y,limit=10)
    assert set([100,101,102]).issubset(index)
    np.testing.assert_allclose(weights[y[index]>5].sum()/weights.sum(),3/103,rtol=1e-6)
    assert not len(set(index))<len(index)


def test_probabilities_are_nested_and_roundtrip_with_no_target_dependency(tmp_path):
    torch.manual_seed(17)
    x=np.array([[1.,2.],[3.,4.]],dtype='float32');rain=np.array([[0.,8.],[12.,30.]],dtype='float32')
    model=CategoricalRain(np.zeros(5),np.ones(5),np.arange(10),width=4)
    prob=model.probabilities(x,rain,[5,10,20])
    assert np.all((prob>=0)&(prob<=1));assert np.all(np.diff(prob,axis=1)<=0)
    data=SimpleNamespace(x=x,rain=rain)
    np.testing.assert_allclose(probabilities(data,model,[5,10,20]),prob)
    model.metadata={'fit_dates':['2021-01-15']};model.save(tmp_path/'head.pt')
    loaded=CategoricalRain.load(tmp_path/'head.pt')
    np.testing.assert_equal(loaded.distribution(x,rain),model.distribution(x,rain))
    np.testing.assert_equal(loaded.rainfall_mean(x,rain),model.rainfall_mean(x,rain))
    with pytest.raises(ValueError,match='boundary'):model.probabilities(x,rain,[7])


def test_head_can_learn_occurrence_and_intensity_from_forecast_features():
    torch.manual_seed(3);torch.set_num_threads(1)
    x=np.repeat(np.array([[-1.],[1.]],dtype='float32'),128,axis=0)
    rain=np.zeros((256,2),dtype='float32');truth=np.r_[np.zeros(128),np.full(128,12.)]
    model=CategoricalRain(np.zeros(4),np.ones(4),np.r_[BOUNDS,60.],width=8)
    z=torch.from_numpy(design(x,rain));y=torch.from_numpy(labels(truth))
    opt=torch.optim.Adam(model.net.parameters(),lr=.03)
    for _ in range(100):
        opt.zero_grad();loss=torch.nn.functional.cross_entropy(model.net(z),y);loss.backward();opt.step()
    p=predict_probabilities(SimpleNamespace(x=x,rain=rain),model,[5,10,20])
    assert p[:128,0].mean()<.02 and p[128:,1].mean()>.98
    assert selection_score(p,truth,[5,10,20])<.001


def test_paired_brier_bootstrap_retains_whole_dates():
    a=np.full(4,.5);b=np.array([0.,1.,0.,1.]);obs=np.array([0.,1.,0.,1.]);day=np.array([0,0,1,1])
    result=paired_brier_interval(a,b,obs,day,2)
    assert result['model_minus_reference']==-.25
    np.testing.assert_allclose(result['day_bootstrap_95_interval'],[-.25,-.25])


def test_paired_amount_bootstrap_compares_errors_on_complete_windows():
    from scripts.check_rain_gauges import paired_amount_interval
    obs=np.ones((48,2));raw=obs*3;model=obs*2
    result=paired_amount_interval(raw,model,obs,6)
    assert result['windows']==16
    np.testing.assert_allclose(result['rmse_difference']['day_bootstrap_95_interval'],[-6,-6])
    assert result['mae_difference']['model_minus_reference']==-6
