from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from biostat_cli.config import (
    PAIRWISE_STATS,
    PairwiseColumns,
    detect_pairwise_columns,
    get_table_config,
    load_resources,
    parse_csv_arg,
    parse_eval_totals,
    parse_stats,
    parse_thresholds,
)
from biostat_cli.evaluators.base import BaseEvaluator, Contingency
from biostat_cli.evaluators.gene import GeneEvaluator, SUM_VARIANTS_SENTINEL
from biostat_cli.evaluators.variant import VariantEvaluator
from biostat_cli.io import scan_table, write_json, write_tsv
from biostat_cli.stats.factory import PairwiseStatOutput, StatFactory

ERROR_INVALID_THRESHOLD = 22


@dataclass(frozen=True)
class RunArgs:
    resources_json: str
    table_name: str
    eval_level: str
    stat: str
    eval_set: str | None
    filters: str | None
    thresholds: str | None
    case_total: float | None
    ctrl_total: float | None
    case_total_by_eval: str | None
    ctrl_total_by_eval: str | None
    out_fname: str
    write_missing: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BioStat CLI")
    parser.add_argument("--resources-json", default="resources.json")
    parser.add_argument("--table-name", required=True)
    parser.add_argument("--eval-level", required=True, choices=["variant", "gene"])
    parser.add_argument("--stat", default="all")
    parser.add_argument("--eval-set", default=None, help="Comma-separated eval columns")
    parser.add_argument("--filters", default=None, help="Comma-separated logical filter names")
    parser.add_argument("--thresholds", default=None, help="Comma-separated percentile thresholds")
    parser.add_argument("--case-total", type=float, default=None)
    parser.add_argument("--ctrl-total", type=float, default=None)
    parser.add_argument(
        "--case-total-by-eval",
        default=None,
        help='Comma-separated per-eval case totals in format "eval_name:value"',
    )
    parser.add_argument(
        "--ctrl-total-by-eval",
        default=None,
        help='Comma-separated per-eval control totals in format "eval_name:value"',
    )
    parser.add_argument("--out-fname", required=True)
    parser.add_argument("--write-missing", choices=["none", "all", "any"], default="none")
    return parser


def _choose_evaluator(eval_level: str, source: pl.LazyFrame) -> BaseEvaluator:
    if eval_level == "variant":
        return VariantEvaluator(source)
    if eval_level == "gene":
        return GeneEvaluator(source)
    raise ValueError(f"Unsupported eval-level: {eval_level}")


def _resolve_eval_cols(raw: str | None, metadata_cols: list[str], eval_level: str) -> list[str]:
    explicit = parse_csv_arg(raw)
    if explicit:
        return explicit
    if metadata_cols:
        return metadata_cols
    if eval_level == "gene":
        # Allow gene weighted mode even when metadata doesn't provide a boolean eval column.
        return [SUM_VARIANTS_SENTINEL]
    raise ValueError("No evaluation columns available from --eval-set or metadata")


def _resolve_filter_cols(raw: str | None, metadata_filters: dict[str, str]) -> list[tuple[str, str | None]]:
    filter_names = parse_csv_arg(raw)
    if not filter_names:
        pairs = list(metadata_filters.items())
    else:
        pairs = []
        for name in filter_names:
            if name.lower() == "none":
                continue
            if name not in metadata_filters:
                raise KeyError(f"Unknown filter name: {name}")
            pairs.append((name, metadata_filters[name]))

    return [("none", None), *pairs]


def _append_binary_row(
    rows: list[dict[str, Any]],
    eval_name: str,
    filter_name: str,
    score_name: str,
    threshold: float,
    stat_name: str,
    value: float,
    p_value: float,
    tp: float,
    fp: float,
    tn: float,
    fn: float,
    rows_used: int,
    total_eval_rows: int,
) -> None:
    rows.append(
        {
            "eval_name": eval_name,
            "filter_name": filter_name,
            "score_name": score_name,
            "threshold": threshold,
            "stat": stat_name,
            "value": value,
            "p_value": p_value,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "rows_used": rows_used,
            "total_eval_rows": total_eval_rows,
        }
    )


def _append_pairwise_row(
    rows: list[dict[str, Any]],
    eval_name: str,
    filter_name: str,
    score_name: str,
    threshold: float,
    out: PairwiseStatOutput,
    cont: Contingency,
    rows_used: int,
    total_eval_rows: int,
) -> None:
    rows.append(
        {
            "eval_name": eval_name,
            "filter_name": filter_name,
            "score_name": score_name,
            "threshold": threshold,
            "stat": out.stat,
            "value": out.value,
            "p_value": out.p_value,
            "anchor_value": out.anchor_value,
            "adjustment_ratio": out.adjustment_ratio,
            "tp": cont.tp,
            "fp": cont.fp,
            "tn": cont.tn,
            "fn": cont.fn,
            "rows_used": rows_used,
            "total_eval_rows": total_eval_rows,
        }
    )


def _resolve_eval_totals(
    eval_col: str,
    table_case_totals: dict[str, float],
    table_ctrl_totals: dict[str, float],
    cli_case_totals: dict[str, float],
    cli_ctrl_totals: dict[str, float],
    global_case_total: float | None,
    global_ctrl_total: float | None,
) -> tuple[float | None, float | None]:
    """
    Resolve denominators for a specific eval column.

    Priority (high -> low):
    1) per-eval CLI override
    2) per-eval values from resources JSON
    3) global CLI --case-total/--ctrl-total
    """
    case_total = cli_case_totals.get(eval_col, table_case_totals.get(eval_col, global_case_total))
    ctrl_total = cli_ctrl_totals.get(eval_col, table_ctrl_totals.get(eval_col, global_ctrl_total))
    return case_total, ctrl_total


def _resolve_output_paths(out_fname: str) -> dict[str, str]:
    base = Path(out_fname)
    if base.suffix:
        base = base.with_suffix("")
    prefix = str(base)
    return {
        "tsv": f"{prefix}.tsv",
        "log": f"{prefix}_log.json",
        "missing_tsv": f"{prefix}_missing.tsv",
    }


def _sort_missing_df(df: pl.DataFrame) -> pl.DataFrame:
    cols = set(df.columns)
    group_sort_cols = [col for col in ["eval_name", "filter_name"] if col in cols]
    cat_sort_expr = (
        pl.when(pl.col("missing_category") == "all_methods")
        .then(pl.lit(0))
        .when(pl.col("missing_category") == "partial_methods")
        .then(pl.lit(1))
        .otherwise(pl.lit(2))
        .alias("__cat_sort")
        if "missing_category" in cols
        else pl.lit(2).alias("__cat_sort")
    )
    if "chrom" in cols and "pos" in cols:
        chrom_norm = pl.col("chrom").cast(pl.Utf8).str.replace(r"(?i)^chr", "").str.to_uppercase()
        chr_num = chrom_norm.cast(pl.Int64, strict=False)
        sorted_df = (
            df.with_columns(
                [
                    cat_sort_expr,
                    pl.when(chr_num.is_not_null())
                    .then(chr_num)
                    .when(chrom_norm == "X")
                    .then(pl.lit(23))
                    .when(chrom_norm == "Y")
                    .then(pl.lit(24))
                    .otherwise(pl.lit(10_000))
                    .alias("__chr_sort"),
                    pl.col("pos").cast(pl.Int64, strict=False).fill_null(9_999_999_999).alias("__pos_sort"),
                ]
            )
            .sort([*group_sort_cols, "__cat_sort", "__chr_sort", "__pos_sort"])
            .drop(["__cat_sort", "__chr_sort", "__pos_sort"])
        )
        return sorted_df

    if "CHROM" in cols and "POS" in cols:
        chrom_norm = pl.col("CHROM").cast(pl.Utf8).str.replace(r"(?i)^chr", "").str.to_uppercase()
        chr_num = chrom_norm.cast(pl.Int64, strict=False)
        sorted_df = (
            df.with_columns(
                [
                    cat_sort_expr,
                    pl.when(chr_num.is_not_null())
                    .then(chr_num)
                    .when(chrom_norm == "X")
                    .then(pl.lit(23))
                    .when(chrom_norm == "Y")
                    .then(pl.lit(24))
                    .otherwise(pl.lit(10_000))
                    .alias("__chr_sort"),
                    pl.col("POS").cast(pl.Int64, strict=False).fill_null(9_999_999_999).alias("__pos_sort"),
                ]
            )
            .sort([*group_sort_cols, "__cat_sort", "__chr_sort", "__pos_sort"])
            .drop(["__cat_sort", "__chr_sort", "__pos_sort"])
        )
        return sorted_df

    for gene_col in ["gene_symbol", "GENE_ID", "gene_id", "ensg"]:
        if gene_col in cols:
            return df.with_columns(cat_sort_expr).sort([*group_sort_cols, "__cat_sort", gene_col]).drop("__cat_sort")
    return df


def _resolve_entity_id_cols(frame: pl.LazyFrame) -> list[str]:
    cols = frame.collect_schema().names()
    candidates = [
        ["chrom", "pos", "ref", "alt"],
        ["CHROM", "POS", "REF", "ALT"],
        ["locus", "alleles"],
        ["GENE_ID"],
        ["gene_id"],
        ["ensg"],
        ["gene_symbol"],
    ]
    for candidate in candidates:
        if all(col in cols for col in candidate):
            return candidate
    fallback = [
        c
        for c in ["chrom", "pos", "ref", "alt", "locus", "alleles", "GENE_ID", "gene_id", "ensg", "gene_symbol"]
        if c in cols
    ]
    if fallback:
        return fallback
    raise ValueError("Unable to infer identifier columns for missing-output report.")


def _build_missing_variant_rows(
    prepared_frame: pl.LazyFrame,
    score_cols: list[str],
    eval_name: str,
    filter_name: str,
    mode: str,
) -> list[dict[str, Any]]:
    id_cols = _resolve_entity_id_cols(prepared_frame)
    missing_aliases = [f"__missing__{score}" for score in score_cols]

    lf = prepared_frame.select(
        [pl.col(c) for c in id_cols]
        + [pl.col(score).is_null().alias(alias) for score, alias in zip(score_cols, missing_aliases)]
    )
    grouped = lf.group_by(id_cols).agg([pl.col(alias).any().alias(alias) for alias in missing_aliases])

    missing_name_exprs = [
        pl.when(pl.col(alias)).then(pl.lit(score)).otherwise(pl.lit(None))
        for score, alias in zip(score_cols, missing_aliases)
    ]
    missing_count_expr = pl.sum_horizontal([pl.col(alias).cast(pl.Int64) for alias in missing_aliases]).alias(
        "missing_score_count"
    )
    all_missing_expr = pl.all_horizontal([pl.col(alias) for alias in missing_aliases])
    any_missing_expr = pl.any_horizontal([pl.col(alias) for alias in missing_aliases])

    report = grouped.with_columns(
        [
            pl.lit(eval_name).alias("eval_name"),
            pl.lit(filter_name).alias("filter_name"),
            missing_count_expr,
            pl.concat_str(missing_name_exprs, separator=",", ignore_nulls=True).alias("missing_score_names"),
            pl.when(all_missing_expr)
            .then(pl.lit("all_methods"))
            .otherwise(pl.lit("partial_methods"))
            .alias("missing_category"),
        ]
    )
    if mode == "all":
        report = report.filter(all_missing_expr)
    else:
        report = report.filter(any_missing_expr)

    out_cols = [
        "eval_name",
        "filter_name",
        *id_cols,
        "missing_category",
        "missing_score_count",
        "missing_score_names",
    ]
    return report.select(out_cols).collect(streaming=True).to_dicts()


def run(args: RunArgs) -> tuple[pl.DataFrame, list[dict[str, Any]], pl.DataFrame]:
    resources = load_resources(args.resources_json)
    table = get_table_config(resources, args.table_name)
    thresholds = parse_thresholds(args.thresholds)
    requested_stats = parse_stats(args.stat)

    source = scan_table(table.path)
    evaluator = _choose_evaluator(args.eval_level, source)

    eval_cols = _resolve_eval_cols(args.eval_set, table.evals, args.eval_level)
    filter_pairs = _resolve_filter_cols(args.filters, table.filters)
    case_totals_by_eval = parse_eval_totals(args.case_total_by_eval, "--case-total-by-eval")
    ctrl_totals_by_eval = parse_eval_totals(args.ctrl_total_by_eval, "--ctrl-total-by-eval")

    # Detect pairwise columns if pairwise stats are requested
    need_pairwise = bool(requested_stats & PAIRWISE_STATS)
    pairwise_cols: PairwiseColumns | None = None
    if need_pairwise:
        table_columns = source.collect_schema().names()
        pairwise_cols = detect_pairwise_columns(table_columns)
        if pairwise_cols is None:
            raise ValueError(
                "Pairwise stats requested but pairwise column structure not detected. "
                "Expected columns: {anchor}_anchor_percentile, {vsm}_percentile_with_anchor, "
                "{anchor}_anchor_percentile_with_{vsm}"
            )

    rows: list[dict[str, Any]] = []
    eval_filter_timings: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for eval_col in eval_cols:
        eval_case_total, eval_ctrl_total = _resolve_eval_totals(
            eval_col=eval_col,
            table_case_totals=table.case_totals,
            table_ctrl_totals=table.ctrl_totals,
            cli_case_totals=case_totals_by_eval,
            cli_ctrl_totals=ctrl_totals_by_eval,
            global_case_total=args.case_total,
            global_ctrl_total=args.ctrl_total,
        )
        for filter_name, filter_col in filter_pairs:
            combo_start = time.perf_counter()
            prepared = evaluator.prepare_eval_frame(eval_col=eval_col, filter_col=filter_col)
            if args.write_missing != "none":
                missing_rows.extend(
                    _build_missing_variant_rows(
                        prepared_frame=prepared.frame,
                        score_cols=table.score_cols,
                        eval_name=eval_col,
                        filter_name=filter_name,
                        mode=args.write_missing,
                    )
                )

            need_labels = "auc" in requested_stats or "auprc" in requested_stats
            need_cont = "enrichment" in requested_stats or "rate_ratio" in requested_stats

            for score_col in table.score_cols:
                score_frame = evaluator.prepare_score_frame(prepared, score_col=score_col)

                if need_labels:
                    labels_scores = evaluator.labels_and_scores(score_frame, eval_col=eval_col, score_col=score_col)
                    labels = labels_scores[0] if labels_scores else None
                    scores = labels_scores[1] if labels_scores else None
                    if "auc" in requested_stats:
                        out = StatFactory.auc(labels, scores)
                        _append_binary_row(
                            rows=rows,
                            eval_name=eval_col,
                            filter_name=filter_name,
                            score_name=score_col,
                            threshold=float("nan"),
                            stat_name=out.stat,
                            value=out.value,
                            p_value=out.p_value,
                            tp=float("nan"),
                            fp=float("nan"),
                            tn=float("nan"),
                            fn=float("nan"),
                            rows_used=score_frame.rows_used,
                            total_eval_rows=prepared.total_eval_rows,
                        )
                    if "auprc" in requested_stats:
                        out = StatFactory.auprc(labels, scores)
                        _append_binary_row(
                            rows=rows,
                            eval_name=eval_col,
                            filter_name=filter_name,
                            score_name=score_col,
                            threshold=float("nan"),
                            stat_name=out.stat,
                            value=out.value,
                            p_value=out.p_value,
                            tp=float("nan"),
                            fp=float("nan"),
                            tn=float("nan"),
                            fn=float("nan"),
                            rows_used=score_frame.rows_used,
                            total_eval_rows=prepared.total_eval_rows,
                        )

                if need_cont and thresholds:
                    conts = evaluator.contingency_batch(
                        score_frame, eval_col=eval_col, score_col=score_col, thresholds=thresholds
                    )
                    if "enrichment" in requested_stats:
                        enr_results = StatFactory.enrichment_batch(conts)
                        for threshold, cont, out in zip(thresholds, conts, enr_results):
                            _append_binary_row(
                                rows=rows,
                                eval_name=eval_col,
                                filter_name=filter_name,
                                score_name=score_col,
                                threshold=threshold,
                                stat_name=out.stat,
                                value=out.value,
                                p_value=out.p_value,
                                tp=cont.tp,
                                fp=cont.fp,
                                tn=cont.tn,
                                fn=cont.fn,
                                rows_used=score_frame.rows_used,
                                total_eval_rows=prepared.total_eval_rows,
                            )
                    if "rate_ratio" in requested_stats:
                        rr_results = StatFactory.rate_ratio_batch(conts, case_total=eval_case_total, ctrl_total=eval_ctrl_total)
                        for threshold, cont, out in zip(thresholds, conts, rr_results):
                            _append_binary_row(
                                rows=rows,
                                eval_name=eval_col,
                                filter_name=filter_name,
                                score_name=score_col,
                                threshold=threshold,
                                stat_name=out.stat,
                                value=out.value,
                                p_value=out.p_value,
                                tp=cont.tp,
                                fp=cont.fp,
                                tn=cont.tn,
                                fn=cont.fn,
                                rows_used=score_frame.rows_used,
                                total_eval_rows=prepared.total_eval_rows,
                            )

            # Pairwise stats: compute adjusted enrichment/rate_ratio for each VSM
            if need_pairwise and pairwise_cols is not None and thresholds:
                # Compute anchor contingency on full set (S* ∩ S_e)
                anchor_score_frame = evaluator.prepare_score_frame(prepared, score_col=pairwise_cols.anchor_full_col)
                anchor_conts_full = evaluator.contingency_batch(
                    anchor_score_frame, eval_col=eval_col, score_col=pairwise_cols.anchor_full_col, thresholds=thresholds
                )

                # For each VSM pair, compute adjusted statistics
                for vsm_base, vsm_col, anchor_pairwise_col in pairwise_cols.vsm_pairs:
                    # Prepare frame for pairwise intersection (filter where both VSM and anchor have values)
                    pairwise_lf = prepared.frame.filter(
                        pl.col(vsm_col).is_not_null() & pl.col(anchor_pairwise_col).is_not_null()
                    )
                    pairwise_df = pairwise_lf.collect(streaming=True)
                    pairwise_rows_used = pairwise_df.height

                    if pairwise_rows_used == 0:
                        continue

                    # Compute VSM contingency on pairwise intersection
                    vsm_conts = evaluator.contingency_batch(
                        evaluator.prepare_score_frame(prepared, score_col=vsm_col),
                        eval_col=eval_col,
                        score_col=vsm_col,
                        thresholds=thresholds,
                    )

                    # Compute anchor contingency on pairwise intersection
                    anchor_conts_pairwise = evaluator.contingency_batch(
                        evaluator.prepare_score_frame(prepared, score_col=anchor_pairwise_col),
                        eval_col=eval_col,
                        score_col=anchor_pairwise_col,
                        thresholds=thresholds,
                    )

                    for threshold, anchor_cont_full, anchor_cont_pw, vsm_cont in zip(
                        thresholds, anchor_conts_full, anchor_conts_pairwise, vsm_conts
                    ):
                        if "pairwise_enrichment" in requested_stats:
                            out = StatFactory.pairwise_enrichment(anchor_cont_full, anchor_cont_pw, vsm_cont)
                            _append_pairwise_row(
                                rows=rows,
                                eval_name=eval_col,
                                filter_name=filter_name,
                                score_name=vsm_base,
                                threshold=threshold,
                                out=out,
                                cont=vsm_cont,
                                rows_used=pairwise_rows_used,
                                total_eval_rows=prepared.total_eval_rows,
                            )
                        if "pairwise_rate_ratio" in requested_stats:
                            out = StatFactory.pairwise_rate_ratio(
                                anchor_cont_full, anchor_cont_pw, vsm_cont, eval_case_total, eval_ctrl_total
                            )
                            _append_pairwise_row(
                                rows=rows,
                                eval_name=eval_col,
                                filter_name=filter_name,
                                score_name=vsm_base,
                                threshold=threshold,
                                out=out,
                                cont=vsm_cont,
                                rows_used=pairwise_rows_used,
                                total_eval_rows=prepared.total_eval_rows,
                            )

                # Also output the anchor VSM itself with adjustment_ratio=1.0
                for threshold, anchor_cont in zip(thresholds, anchor_conts_full):
                    if "pairwise_enrichment" in requested_stats:
                        out = StatFactory.pairwise_enrichment(anchor_cont, anchor_cont, anchor_cont)
                        _append_pairwise_row(
                            rows=rows,
                            eval_name=eval_col,
                            filter_name=filter_name,
                            score_name=pairwise_cols.anchor_base,
                            threshold=threshold,
                            out=out,
                            cont=anchor_cont,
                            rows_used=anchor_score_frame.rows_used,
                            total_eval_rows=prepared.total_eval_rows,
                        )
                    if "pairwise_rate_ratio" in requested_stats:
                        out = StatFactory.pairwise_rate_ratio(
                            anchor_cont, anchor_cont, anchor_cont, eval_case_total, eval_ctrl_total
                        )
                        _append_pairwise_row(
                            rows=rows,
                            eval_name=eval_col,
                            filter_name=filter_name,
                            score_name=pairwise_cols.anchor_base,
                            threshold=threshold,
                            out=out,
                            cont=anchor_cont,
                            rows_used=anchor_score_frame.rows_used,
                            total_eval_rows=prepared.total_eval_rows,
                        )

            eval_filter_timings.append(
                {
                    "eval_name": eval_col,
                    "filter_name": filter_name,
                    "elapsed_seconds": time.perf_counter() - combo_start,
                }
            )
    if missing_rows:
        missing_df = _sort_missing_df(pl.DataFrame(missing_rows))
    else:
        missing_df = pl.DataFrame(
            schema={
                "eval_name": pl.String,
                "filter_name": pl.String,
                "missing_category": pl.String,
                "missing_score_count": pl.Int64,
                "missing_score_names": pl.String,
            }
        )
    return pl.DataFrame(rows), eval_filter_timings, missing_df


def main() -> None:
    parser = _build_parser()
    ns = parser.parse_args()
    args = RunArgs(
        resources_json=ns.resources_json,
        table_name=ns.table_name,
        eval_level=ns.eval_level,
        stat=ns.stat,
        eval_set=ns.eval_set,
        filters=ns.filters,
        thresholds=ns.thresholds,
        case_total=ns.case_total,
        ctrl_total=ns.ctrl_total,
        case_total_by_eval=ns.case_total_by_eval,
        ctrl_total_by_eval=ns.ctrl_total_by_eval,
        out_fname=ns.out_fname,
        write_missing=ns.write_missing,
    )
    try:
        output_paths = _resolve_output_paths(args.out_fname)
        start = time.perf_counter()
        out, eval_filter_timings, missing_df = run(args)
        write_tsv(out, output_paths["tsv"])
        if args.write_missing != "none":
            write_tsv(missing_df, output_paths["missing_tsv"])
        elapsed_seconds = time.perf_counter() - start
        write_json(
            {
                "run_args": asdict(args),
                "table_path": get_table_config(load_resources(args.resources_json), args.table_name).path,
                "output_files": output_paths,
                "elapsed_seconds": elapsed_seconds,
                "eval_filter_elapsed_seconds": eval_filter_timings,
            },
            output_paths["log"],
        )
        print(f"Elapsed time (s): {elapsed_seconds:.3f}")
        for item in eval_filter_timings:
            print(
                f"  eval={item['eval_name']} filter={item['filter_name']} "
                f"time_s={item['elapsed_seconds']:.3f}"
            )
    except ValueError as exc:
        if "threshold" in str(exc).lower():
            print(f"Error [{ERROR_INVALID_THRESHOLD}]: {exc}", file=sys.stderr)
            raise SystemExit(ERROR_INVALID_THRESHOLD) from exc
        raise


if __name__ == "__main__":
    main()
