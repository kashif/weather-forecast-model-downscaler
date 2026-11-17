"""Rare-event discrimination and alert rules selected without test labels."""
from __future__ import annotations
import numpy as np


def counts_to_scores(tp,fp,fn,tn):
    def ratio(a,b):return float(a/b) if b else None
    total=tp+fp+fn+tn
    return dict(hits=int(tp),false_alarms=int(fp),misses=int(fn),correct_negatives=int(tn),
                csi=ratio(tp,tp+fp+fn),precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),
                false_positive_rate=ratio(fp,fp+tn),false_alarm_ratio=ratio(fp,tp+fp),
                alert_fraction=ratio(tp+fp,total),observed_events=int(tp+fn))


def contingency(probability,observed,cutoff):
    probability=np.asarray(probability);observed=np.asarray(observed,dtype=bool)
    if probability.shape!=observed.shape:raise ValueError('probability/label shape differs')
    if not np.isfinite(probability).all() or ((probability<0)|(probability>1)).any():raise ValueError('invalid probabilities')
    alert=np.zeros_like(observed) if cutoff is None else probability>=cutoff
    return counts_to_scores(np.count_nonzero(alert&observed),np.count_nonzero(alert&~observed),
                            np.count_nonzero(~alert&observed),np.count_nonzero(~alert&~observed))


def select_rules(probability,observed,cutoffs,false_positive_budget):
    """CSI and false-positive-budget rules; all ties prefer the higher cutoff."""
    rows=[dict(cutoff=float(c),**contingency(probability,observed,c)) for c in cutoffs]
    if not np.any(observed):
        return dict(max_csi=None,fpr_budget=None,note='no selection positives: no-alert policy',sweep=rows)
    best=max(rows,key=lambda r:((r['csi'] or 0),r['cutoff']))
    allowed=[r for r in rows if r['false_positive_rate'] is not None and r['false_positive_rate']<=false_positive_budget]
    budget=max(allowed,key=lambda r:((r['recall'] or 0),r['cutoff'])) if allowed else None
    return dict(max_csi=best['cutoff'],fpr_budget=budget['cutoff'] if budget else None,
                note='cutoffs selected on validation only; ties prefer fewer alerts',sweep=rows)


def precision_recall(probability,observed,false_positive_budget=.001):
    """Exact tie-grouped average precision plus a compact PR curve.

    Recall-at-FPR is a curve diagnostic, not a deployable cutoff selected on test.
    """
    probability=np.asarray(probability,dtype='float64');observed=np.asarray(observed,dtype=bool)
    if probability.ndim!=1 or observed.ndim!=1:raise ValueError('event curves require one-dimensional arrays')
    contingency(probability,observed,None)  # Validate even degenerate/no-positive inputs.
    if not len(observed):raise ValueError('empty event data')
    order=np.argsort(-probability,kind='stable');score=probability[order];labels=observed[order]
    ends=np.r_[np.flatnonzero(score[:-1]!=score[1:]),len(score)-1]
    tp=labels.cumsum(dtype='int64')[ends];fp=ends+1-tp
    positives=int(observed.sum());negatives=len(observed)-positives
    precision=tp/(ends+1)
    recall=tp/positives if positives else np.zeros_like(precision)
    fpr=fp/negatives if negatives else np.zeros_like(precision)
    ap=float(np.sum(np.diff(np.r_[0.,recall])*precision)) if positives else None
    # Retain recall-spaced points and log-spaced early ranks, not millions of rows.
    keep=np.unique(np.r_[np.searchsorted(recall,np.linspace(0,1,301)),
                         np.geomspace(1,len(ends),min(len(ends),301)).astype(int)-1,len(ends)-1])
    keep=keep[keep<len(ends)]
    valid=np.flatnonzero(fpr<=false_positive_budget)
    best=int(valid[np.argmax(recall[valid])]) if len(valid) and positives and negatives else None
    diagnostic=None if best is None else dict(recall=float(recall[best]),precision=float(precision[best]),
                                              actual_fpr=float(fpr[best]),cutoff=float(score[ends[best]]))
    return dict(average_precision=ap,prevalence=positives/len(observed),positive_points=positives,
                points=len(observed),curve=[dict(recall=float(recall[i]),precision=float(precision[i]),
                                                cutoff=float(score[ends[i]]),fpr=float(fpr[i])) for i in keep],
                recall_at_fpr_curve_diagnostic=diagnostic,
                diagnostic_note='curve summary using labels; do not use its cutoff as a test-selected alert rule')


def paired_day_csi_interval(raw_days,model_days,seed=17):
    if len(raw_days)!=len(model_days) or not raw_days:raise ValueError('matching days required')
    if [d.get('date') for d in raw_days]!=[d.get('date') for d in model_days]:raise ValueError('day pairing differs')
    rng=np.random.default_rng(seed);index=rng.integers(0,len(raw_days),size=(10000,len(raw_days)))
    def draws(days):
        values=np.array([[d['hits'],d['false_alarms'],d['misses']] for d in days])
        totals=values[index].sum(1);denominator=totals.sum(1)
        return np.divide(totals[:,0],denominator,out=np.full(len(totals),np.nan),where=denominator>0)
    delta=draws(model_days)-draws(raw_days);valid=delta[np.isfinite(delta)]
    return dict(day_bootstrap_95_interval=np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,
                days=len(raw_days),warning='Only four test days; descriptive resampling, not general storm-skill evidence')
