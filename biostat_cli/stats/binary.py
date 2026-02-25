from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import poisson

from biostat_cli.evaluators.base import Contingency


@dataclass(frozen=True)
class BinaryStatResult:
    value: float
    p_value: float


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return math.nan
    return num / den


def enrichment(cont: Contingency) -> BinaryStatResult:
    case_rate = _safe_div(cont.tp, cont.tp + cont.fn)
    ctrl_rate = _safe_div(cont.fp, cont.fp + cont.tn)
    value = _safe_div(case_rate, ctrl_rate) if not math.isnan(case_rate) and not math.isnan(ctrl_rate) else math.nan
    return BinaryStatResult(value=value, p_value=poisson_p_value(cont))


def rate_ratio(cont: Contingency, case_total: float | None, ctrl_total: float | None) -> BinaryStatResult:
    if case_total is None or ctrl_total is None:
        return BinaryStatResult(value=math.nan, p_value=poisson_p_value(cont))
    case_rate = _safe_div(cont.tp, case_total)
    ctrl_rate = _safe_div(cont.fp, ctrl_total)
    value = _safe_div(case_rate, ctrl_rate) if not math.isnan(case_rate) and not math.isnan(ctrl_rate) else math.nan
    return BinaryStatResult(value=value, p_value=poisson_p_value(cont))


def poisson_p_value(cont: Contingency) -> float:
    above_pos = cont.tp
    above_total = cont.tp + cont.fp
    below_pos = cont.fn
    below_total = cont.fn + cont.tn
    if above_total <= 0 or below_total <= 0:
        return math.nan

    below_rate = below_pos / below_total
    expected = below_rate * above_total
    return float(poisson.sf(above_pos - 1, expected))
