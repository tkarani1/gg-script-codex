import math

from biostat_cli.evaluators.base import Contingency
from biostat_cli.stats.binary import enrichment, rate_ratio
from biostat_cli.stats.continuous import compute_auc, compute_auprc


def test_auc_and_auprc_basic():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert compute_auc(labels, scores) > 0.99
    assert compute_auprc(labels, scores) > 0.99


def test_binary_stats():
    cont = Contingency(tp=10, fp=5, tn=20, fn=15)
    enr = enrichment(cont)
    rr = rate_ratio(cont, case_total=200, ctrl_total=300)
    assert not math.isnan(enr.value)
    assert not math.isnan(enr.p_value)
    assert not math.isnan(rr.value)
    assert not math.isnan(rr.p_value)
