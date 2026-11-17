"""Alert evaluation distinguishes calibration, ranking, and frozen decisions."""
import numpy as np
import pytest
from downscale.rain_alerts import contingency,precision_recall,select_rules,paired_day_csi_interval


def test_tied_average_precision_has_no_order_dependence():
    score=np.array([.9,.8,.8,.1]);obs=np.array([1,0,1,0],dtype=bool)
    result=precision_recall(score,obs)
    assert result['average_precision']==pytest.approx(5/6)
    order=np.array([0,2,1,3])
    assert precision_recall(score[order],obs[order])['average_precision']==pytest.approx(5/6)
    assert precision_recall(np.zeros(4),obs)['average_precision']==pytest.approx(.5)


def test_selected_cutoff_is_applied_unchanged_to_new_labels():
    score=np.array([.1,.2,.3,.4]);selection=np.array([0,0,1,1],dtype=bool)
    rules=select_rules(score,selection,[.1,.2,.3,.5],.01)
    assert rules['max_csi']==.3 and rules['fpr_budget']==.3
    fresh=np.array([1,0,1,0],dtype=bool)
    result=contingency(score,fresh,rules['max_csi'])
    assert result['hits']==1 and result['false_alarms']==1 and result['misses']==1
    assert result['csi']==pytest.approx(1/3)
    assert result['false_positive_rate']==.5


def test_false_positive_rate_differs_from_false_alarm_ratio():
    prob=np.array([.9,.8,.1,.1,.1]);obs=np.array([1,0,0,0,0],dtype=bool)
    r=contingency(prob,obs,.5)
    assert r['false_positive_rate']==.25
    assert r['false_alarm_ratio']==.5
    assert contingency(prob,obs,None)['alert_fraction']==0


def test_empty_positive_support_does_not_choose_a_test_based_rule():
    prob=np.array([.1,.2,.9]);obs=np.zeros(3,dtype=bool)
    rules=select_rules(prob,obs,[.1,.5,1.],.001)
    assert rules['max_csi'] is None and rules['fpr_budget'] is None
    assert precision_recall(prob,obs)['average_precision'] is None
    assert contingency(prob,obs,None)['csi'] is None
    with pytest.raises(ValueError):precision_recall(np.array([np.nan]),np.array([True]))


def test_day_bootstrap_compares_corresponding_day_counts():
    a=[dict(hits=2,false_alarms=1,misses=1),dict(hits=4,false_alarms=1,misses=1)]
    result=paired_day_csi_interval(a,a)
    assert result['day_bootstrap_95_interval']==[0.,0.]


def test_monotone_probability_change_preserves_ranking_not_fixed_cutoff():
    p=np.array([.1,.2,.4,.7]);obs=np.array([0,1,1,0],dtype=bool)
    q=np.sqrt(p)
    assert precision_recall(p,obs)['average_precision']==precision_recall(q,obs)['average_precision']
    assert contingency(p,obs,.5)['hits']!=contingency(q,obs,.5)['hits']
    with pytest.raises(ValueError,match='one-dimensional'):
        precision_recall(p[:,None],obs[:,None])
    with pytest.raises(ValueError,match='pairing'):
        paired_day_csi_interval([dict(date='a',hits=1,false_alarms=0,misses=1)],
                                [dict(date='b',hits=1,false_alarms=0,misses=1)])
