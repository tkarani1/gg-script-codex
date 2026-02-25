from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any

import polars as pl

from biostat_cli.config import get_table_config, load_resources, parse_csv_arg, parse_stats, parse_thresholds
from biostat_cli.evaluators.base import BaseEvaluator
from biostat_cli.evaluators.gene import GeneEvaluator, SUM_VARIANTS_SENTINEL
from biostat_cli.evaluators.variant import VariantEvaluator
from biostat_cli.io import scan_table, write_json, write_tsv
from biostat_cli.stats.factory import StatFactory


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
    output_tsv: str
    output_log: str


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
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--output-log", required=True)
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


def run(args: RunArgs) -> pl.DataFrame:
    resources = load_resources(args.resources_json)
    table = get_table_config(resources, args.table_name)
    thresholds = parse_thresholds(args.thresholds)
    requested_stats = parse_stats(args.stat)

    source = scan_table(table.path)
    evaluator = _choose_evaluator(args.eval_level, source)

    eval_cols = _resolve_eval_cols(args.eval_set, table.evals, args.eval_level)
    filter_pairs = _resolve_filter_cols(args.filters, table.filters)

    rows: list[dict[str, Any]] = []

    for eval_col in eval_cols:
        for filter_name, filter_col in filter_pairs:
            prepared = evaluator.prepare_eval_frame(eval_col=eval_col, filter_col=filter_col)

            for score_col in table.score_cols:
                score_frame = evaluator.prepare_score_frame(prepared, score_col=score_col)
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

                for threshold in thresholds:
                    cont = evaluator.contingency(score_frame, eval_col=eval_col, score_col=score_col, threshold=threshold)
                    if "enrichment" in requested_stats:
                        out = StatFactory.enrichment(cont)
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
                        out = StatFactory.rate_ratio(
                            cont=cont,
                            case_total=args.case_total,
                            ctrl_total=args.ctrl_total,
                        )
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
    return pl.DataFrame(rows)


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
        output_tsv=ns.output_tsv,
        output_log=ns.output_log,
    )
    out = run(args)
    write_tsv(out, args.output_tsv)
    write_json({"run_args": asdict(args)}, args.output_log)


if __name__ == "__main__":
    main()
