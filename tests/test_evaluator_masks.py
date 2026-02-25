import polars as pl

from biostat_cli.evaluators.variant import VariantEvaluator


def test_total_eval_rows_after_eval_drop_and_filter():
    df = pl.DataFrame(
        {
            "score": [10.0, 20.0, None, 40.0],
            "eval_col": [True, False, None, True],
            "filt": [True, False, True, True],
        }
    )
    ev = VariantEvaluator(df.lazy())
    prepared = ev.prepare_eval_frame(eval_col="eval_col", filter_col="filt")
    # Rows 0 and 3 survive eval non-null + filter True.
    assert prepared.total_eval_rows == 2

    score_frame = ev.prepare_score_frame(prepared, score_col="score")
    assert score_frame.rows_used == 2
