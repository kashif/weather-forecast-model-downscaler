import numpy as np

from downscale.rain_alerts import contingency
from scripts.calibrate_gauge_alerts import cutoff_for_budget


def test_frozen_budget_cutoff_respects_ties():
    probability = np.array([.9, .9, .8, .2, .95])
    observed = np.array([False, False, False, False, True])
    cutoff = cutoff_for_budget(probability, observed, .25)
    scores = contingency(probability, observed, cutoff)
    assert scores['false_alarms'] == 0
    assert scores['hits'] == 1


def test_zero_budget_with_certain_negative_produces_no_alert_rule():
    cutoff = cutoff_for_budget(np.array([1., 1.]), np.array([False, True]), 0.)
    assert cutoff is None
