from __future__ import annotations

import polars as pl

from biostat_cli.evaluators.base import BaseEvaluator, Contingency, ScoreFrame


class VariantEvaluator(BaseEvaluator):
    def contingency(self, score_frame: ScoreFrame, eval_col: str, score_col: str, threshold: float) -> Contingency:
        above = pl.col(score_col) > threshold
        is_pos = pl.col(eval_col) == True  # noqa: E712
        is_neg = pl.col(eval_col) == False  # noqa: E712

        out = score_frame.frame.select(
            pl.when(above & is_pos).then(1).otherwise(0).sum().cast(pl.Float64).alias("tp"),
            pl.when(above & is_neg).then(1).otherwise(0).sum().cast(pl.Float64).alias("fp"),
            pl.when((~above) & is_neg).then(1).otherwise(0).sum().cast(pl.Float64).alias("tn"),
            pl.when((~above) & is_pos).then(1).otherwise(0).sum().cast(pl.Float64).alias("fn"),
        ).collect(streaming=True)
        row = out.to_dicts()[0]
        return Contingency(tp=row["tp"], fp=row["fp"], tn=row["tn"], fn=row["fn"])

    def labels_and_scores(
        self, score_frame: ScoreFrame, eval_col: str, score_col: str
    ) -> tuple[list[int], list[float]] | None:
        out = score_frame.frame.select(
            pl.col(eval_col).cast(pl.Int64).alias("label"),
            pl.col(score_col).cast(pl.Float64).alias("score"),
        ).collect(streaming=True)
        labels = out["label"].to_list()
        scores = out["score"].to_list()
        return labels, scores
