from __future__ import annotations

import math
from dataclasses import dataclass

from biostat_cli.evaluators.base import Contingency
from biostat_cli.stats.binary import enrichment, rate_ratio
from biostat_cli.stats.continuous import compute_auc, compute_auprc


@dataclass(frozen=True)
class StatOutput:
    stat: str
    value: float
    p_value: float


class StatFactory:
    @staticmethod
    def auc(labels: list[int] | None, scores: list[float] | None) -> StatOutput:
        if labels is None or scores is None:
            return StatOutput(stat="auc", value=math.nan, p_value=math.nan)
        return StatOutput(stat="auc", value=compute_auc(labels, scores), p_value=math.nan)

    @staticmethod
    def auprc(labels: list[int] | None, scores: list[float] | None) -> StatOutput:
        if labels is None or scores is None:
            return StatOutput(stat="auprc", value=math.nan, p_value=math.nan)
        return StatOutput(stat="auprc", value=compute_auprc(labels, scores), p_value=math.nan)

    @staticmethod
    def enrichment(cont: Contingency) -> StatOutput:
        out = enrichment(cont)
        return StatOutput(stat="enrichment", value=out.value, p_value=out.p_value)

    @staticmethod
    def rate_ratio(cont: Contingency, case_total: float | None, ctrl_total: float | None) -> StatOutput:
        out = rate_ratio(cont, case_total=case_total, ctrl_total=ctrl_total)
        return StatOutput(stat="rate_ratio", value=out.value, p_value=out.p_value)
