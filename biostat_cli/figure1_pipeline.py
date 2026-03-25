from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import polars as pl

from biostat_cli import cli as biostat_cli
from biostat_cli.config import detect_pairwise_columns, get_table_config, load_resources
from biostat_cli.io import write_json, write_tsv

FIGURE1_RAW_TABLE_KEY = "FIGURE1_RAW"
FIGURE1_PAIRWISE_TABLE_KEY = "FIGURE1_PAIRWISE"


@dataclass(frozen=True)
class TableRunConfig:
    resources_json: str
    table_name: str


@dataclass(frozen=True)
class PipelineConfig:
    raw_score_columns: list[str]
    pairwise_score_columns: list[str]
    eval_set: list[str]
    panel_order: list[str]
    panel_eval_map: dict[str, str]
    panel_titles: dict[str, str]
    panel_metrics: dict[str, dict[str, str]]
    method_display_names: dict[str, str]
    method_order: list[str]
    rate_ratio_denominators: dict[str, dict[str, float]]
    default_threshold: float
    thresholds: list[float]
    default_filter_name: str


@dataclass(frozen=True)
class PipelineArgs:
    command: str
    config: str
    mode: str
    profile: str
    output_layout: str
    outdir: str | None
    threshold: float | None
    thresholds: str | None
    bootstrap: int | None
    raw_parquet: str | None
    pairwise_parquet: str | None
    panel_table: str | None
    raw_metrics_tsv: str | None
    pairwise_metrics_tsv: str | None
    eval_set_override: str | None
    paper_strict: bool
    overwrite: bool
    dry_run: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Figure 1 pipeline (compute + QC + plotting)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", default="figure1_pipeline_config.json")
        subparser.add_argument("--mode", choices=["raw", "pairwise", "both"], default="both")
        subparser.add_argument(
            "--profile",
            choices=["paper_figure1", "all_variant"],
            default="paper_figure1",
            help="paper_figure1: fixed 4-panel paper match; all_variant: auto-include all is_pos_* variant evals.",
        )
        subparser.add_argument(
            "--output-layout",
            choices=["combined", "per_eval", "both"],
            default="combined",
            help="combined: single aggregated outputs; per_eval: one-file-per-eval; both: write both.",
        )
        subparser.add_argument("--outdir", default=None)
        subparser.add_argument(
            "--threshold", type=float, default=None, help="Threshold used for panel table and plotting."
        )
        subparser.add_argument("--thresholds", default=None, help="Comma-separated compute thresholds override.")
        subparser.add_argument(
            "--bootstrap",
            type=int,
            default=None,
            help="Enable bootstrap std_error with N samples (e.g., 100).",
        )
        subparser.add_argument(
            "--eval-set",
            default=None,
            help="Optional override: comma-separated eval set to force a custom subset.",
        )
        subparser.add_argument(
            "--paper-strict",
            action="store_true",
            help="Fail if any required paper eval is missing (instead of fallback/warning).",
        )
        subparser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing output directory contents.",
        )
        subparser.add_argument("--dry-run", action="store_true")

    def add_parquet_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--raw-parquet",
            default=None,
            help="Parquet for raw (all-VSM) metrics. Required when --mode is raw or both.",
        )
        subparser.add_argument(
            "--pairwise-parquet",
            default=None,
            help="Parquet for pairwise MPC-anchor metrics. Required when --mode is pairwise or both.",
        )

    run_parser = subparsers.add_parser("run", help="Compute metrics, build panel table/QC, and render plots.")
    add_common_args(run_parser)
    add_parquet_args(run_parser)
    compute_parser = subparsers.add_parser("compute", help="Compute metrics and build panel table/QC only.")
    add_common_args(compute_parser)
    add_parquet_args(compute_parser)

    plot_parser = subparsers.add_parser("plot", help="Render plots from an existing panel table TSV.")
    add_common_args(plot_parser)
    plot_parser.add_argument("--panel-table", default=None, help="Existing panel_table.tsv to plot from.")
    plot_parser.add_argument(
        "--raw-metrics-tsv",
        default=None,
        help="Existing raw biostat metrics TSV (from biostat_cli) to build panel table and plot.",
    )
    plot_parser.add_argument(
        "--pairwise-metrics-tsv",
        default=None,
        help="Existing pairwise biostat metrics TSV (from biostat_cli) to build panel table and plot.",
    )
    return parser


def _parse_thresholds(raw: str | None) -> list[float] | None:
    if raw is None or not raw.strip():
        return None
    return [float(piece.strip()) for piece in raw.split(",") if piece.strip()]


def _discover_is_pos_evals(parquet_path: str) -> list[str]:
    """Discover all is_pos_* columns in a parquet file (for all_variant profile)."""
    if parquet_path.startswith("gs://"):
        return []
    cols = pl.scan_parquet(parquet_path).collect_schema().names()
    return sorted([c for c in cols if c.startswith("is_pos_")])


def _resolve_eval_set(
    args: PipelineArgs,
    config: PipelineConfig,
) -> tuple[list[str], dict[str, str], dict[str, str], dict[str, dict[str, str]], list[str]]:
    """
    Resolve eval set based on profile, returning:
    - eval_set: list of eval column names
    - panel_eval_map: {panel_id -> eval_name}
    - panel_titles: {panel_id -> title}
    - panel_metrics: {panel_id -> {raw: stat, pairwise: stat}}
    - warnings: list of warning messages
    """
    warnings: list[str] = []

    if args.eval_set_override:
        evals = [e.strip() for e in args.eval_set_override.split(",") if e.strip()]
        panel_eval_map = {e: e for e in evals}
        panel_titles = {e: e.replace("is_pos_", "").replace("_", " ").title() for e in evals}
        panel_metrics = {
            e: {"raw": config.panel_metrics.get(list(config.panel_metrics.keys())[0], {}).get("raw", "enrichment"),
                "pairwise": config.panel_metrics.get(list(config.panel_metrics.keys())[0], {}).get("pairwise", "pairwise_enrichment")}
            for e in evals
        }
        return evals, panel_eval_map, panel_titles, panel_metrics, warnings

    if args.profile == "paper_figure1":
        return (
            config.eval_set,
            config.panel_eval_map,
            config.panel_titles,
            config.panel_metrics,
            warnings,
        )

    discovered: list[str] = []
    if args.raw_parquet:
        discovered.extend(_discover_is_pos_evals(args.raw_parquet))
    if args.pairwise_parquet:
        for e in _discover_is_pos_evals(args.pairwise_parquet):
            if e not in discovered:
                discovered.append(e)
    discovered = sorted(set(discovered))

    if not discovered:
        warnings.append("all_variant profile: no is_pos_* columns found; falling back to config eval_set.")
        return (
            config.eval_set,
            config.panel_eval_map,
            config.panel_titles,
            config.panel_metrics,
            warnings,
        )

    default_raw_stat = "enrichment"
    default_pairwise_stat = "pairwise_enrichment"
    if config.panel_metrics:
        first_panel = list(config.panel_metrics.values())[0]
        default_raw_stat = first_panel.get("raw", default_raw_stat)
        default_pairwise_stat = first_panel.get("pairwise", default_pairwise_stat)

    panel_eval_map = {e: e for e in discovered}
    panel_titles = {e: e.replace("is_pos_", "").replace("_", " ").title() for e in discovered}
    panel_metrics = {e: {"raw": default_raw_stat, "pairwise": default_pairwise_stat} for e in discovered}

    return discovered, panel_eval_map, panel_titles, panel_metrics, warnings


def _parquet_path_for_resources(path: str) -> str:
    if path.startswith("gs://"):
        return path
    return str(Path(path).expanduser().resolve())


def _materialize_run_resources(
    outdir: str,
    mode: str,
    *,
    raw_parquet: str | None,
    pairwise_parquet: str | None,
    raw_score_columns: list[str],
    pairwise_score_columns: list[str],
) -> tuple[str, TableRunConfig | None, TableRunConfig | None]:
    resources_path = str(Path(outdir) / "figure1_run_resources.json")
    table_info: dict[str, Any] = {}
    if mode in {"raw", "both"}:
        assert raw_parquet is not None
        table_info[FIGURE1_RAW_TABLE_KEY] = {
            "Path": _parquet_path_for_resources(raw_parquet),
            "Level": "variant",
            "Score_cols": list(raw_score_columns),
        }
    if mode in {"pairwise", "both"}:
        assert pairwise_parquet is not None
        table_info[FIGURE1_PAIRWISE_TABLE_KEY] = {
            "Path": _parquet_path_for_resources(pairwise_parquet),
            "Level": "variant",
            "Score_cols": list(pairwise_score_columns),
        }
    write_json({"Table_info": table_info}, resources_path)
    raw_cfg = (
        TableRunConfig(resources_json=resources_path, table_name=FIGURE1_RAW_TABLE_KEY)
        if mode in {"raw", "both"}
        else None
    )
    pw_cfg = (
        TableRunConfig(resources_json=resources_path, table_name=FIGURE1_PAIRWISE_TABLE_KEY)
        if mode in {"pairwise", "both"}
        else None
    )
    return resources_path, raw_cfg, pw_cfg


def _load_pipeline_config(config_path: str) -> tuple[PipelineConfig, str]:
    cfg_path = Path(config_path).resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if "raw_score_columns" not in payload or "pairwise_score_columns" not in payload:
        raise ValueError(
            f"Config {config_path} must define `raw_score_columns` and `pairwise_score_columns` "
            "(table paths are passed via --raw-parquet / --pairwise-parquet on run/compute)."
        )

    config = PipelineConfig(
        raw_score_columns=list(payload["raw_score_columns"]),
        pairwise_score_columns=list(payload["pairwise_score_columns"]),
        eval_set=list(payload["eval_set"]),
        panel_order=list(payload["panel_order"]),
        panel_eval_map=dict(payload["panel_eval_map"]),
        panel_titles=dict(payload["panel_titles"]),
        panel_metrics={
            str(panel_id): {"raw": str(v["raw"]), "pairwise": str(v["pairwise"])}
            for panel_id, v in dict(payload["panel_metrics"]).items()
        },
        method_display_names={str(k): str(v) for k, v in dict(payload["method_display_names"]).items()},
        method_order=list(payload["method_order"]),
        rate_ratio_denominators={
            str(k): {
                kk: float(vv)
                for kk, vv in dict(v).items()
                if vv is not None and kk in {"case_total", "ctrl_total"}
            }
            for k, v in dict(payload["rate_ratio_denominators"]).items()
        },
        default_threshold=float(payload.get("default_threshold", 0.95)),
        thresholds=[float(v) for v in payload.get("thresholds", [0.90, 0.95, 0.98, 0.99])],
        default_filter_name=str(payload.get("default_filter_name", "none")),
    )
    return config, str(cfg_path)


def _format_totals_arg(denominators: dict[str, dict[str, float]], key: str) -> str | None:
    pieces: list[str] = []
    for eval_name, eval_totals in denominators.items():
        if key in eval_totals:
            pieces.append(f"{eval_name}:{eval_totals[key]}")
    return ",".join(pieces) if pieces else None


def _default_outdir(args: PipelineArgs, config_path: str) -> str:
    cfg = Path(config_path).stem
    mode = args.mode
    cmd = args.command
    threshold = args.threshold if args.threshold is not None else "default"
    thresholds = args.thresholds if args.thresholds else "cfg"
    bootstrap = args.bootstrap if args.bootstrap is not None else "none"
    name = f"figure1_{cfg}_{cmd}_{mode}_thr-{threshold}_thrs-{thresholds}_boot-{bootstrap}"
    safe_name = (
        name.replace(",", "-")
        .replace("/", "-")
        .replace(" ", "")
        .replace(":", "-")
    )
    return str((Path.cwd() / "results" / safe_name).resolve())


def _prepare_outdir(outdir: str, overwrite: bool) -> None:
    out_path = Path(outdir)
    if out_path.exists():
        has_existing = any(out_path.iterdir())
        if has_existing and not overwrite:
            raise ValueError(
                f"Output directory already exists and is not empty: {outdir}. "
                "Use --overwrite to replace existing results."
            )
        if has_existing and overwrite:
            shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)


def _validate_config(
    config: PipelineConfig,
    mode: str,
    *,
    raw_parquet: str | None,
    pairwise_parquet: str | None,
    eval_set: list[str],
    panel_eval_map: dict[str, str],
    panel_titles: dict[str, str],
    panel_metrics: dict[str, dict[str, str]],
    paper_strict: bool = False,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    runs: list[tuple[str, str, list[str]]] = []
    if mode in {"raw", "both"}:
        assert raw_parquet is not None
        runs.append(("raw", raw_parquet, config.raw_score_columns))
    if mode in {"pairwise", "both"}:
        assert pairwise_parquet is not None
        runs.append(("pairwise", pairwise_parquet, config.pairwise_score_columns))

    for run_name, table_path, score_cols in runs:
        if table_path.startswith("gs://"):
            warnings.append(f"[{run_name}] table path is remote ({table_path}); local schema checks skipped.")
            continue
        parquet_path = Path(table_path)
        if not parquet_path.exists():
            errors.append(f"[{run_name}] parquet path does not exist: {table_path}")
            continue

        cols = pl.scan_parquet(table_path).collect_schema().names()
        missing_evals = [e for e in eval_set if e not in cols]
        if missing_evals:
            if paper_strict:
                errors.append(f"[{run_name}] missing eval columns (--paper-strict): {missing_evals}")
            else:
                warnings.append(f"[{run_name}] missing eval columns (will skip): {missing_evals}")
        missing_scores = [s for s in score_cols if s not in cols]
        if missing_scores:
            errors.append(f"[{run_name}] missing score columns: {missing_scores}")

        if run_name == "pairwise":
            pairwise_cols = detect_pairwise_columns(cols)
            if pairwise_cols is None:
                errors.append("[pairwise] pairwise score column structure not detected.")

    panel_order = list(panel_eval_map.keys())
    for panel_id in panel_order:
        if panel_id not in panel_eval_map:
            errors.append(f"panel `{panel_id}` missing from panel_eval_map.")
        if panel_id not in panel_titles:
            errors.append(f"panel `{panel_id}` missing from panel_titles.")
        if panel_id not in panel_metrics:
            errors.append(f"panel `{panel_id}` missing from panel_metrics.")

    rate_ratio_required_evals: set[str] = set()
    for panel_id, eval_name in panel_eval_map.items():
        metrics = panel_metrics.get(panel_id, {})
        raw_metric = str(metrics.get("raw", ""))
        pairwise_metric = str(metrics.get("pairwise", ""))
        if "rate_ratio" in raw_metric or "rate_ratio" in pairwise_metric:
            rate_ratio_required_evals.add(eval_name)

    for eval_name in sorted(rate_ratio_required_evals):
        totals = config.rate_ratio_denominators.get(eval_name, {})
        if "case_total" not in totals or "ctrl_total" not in totals:
            warnings.append(
                f"rate-ratio denominators missing/incomplete for `{eval_name}`. "
                "Rate-ratio values for that eval will be NaN."
            )
    return errors, warnings


def _run_biostat(
    *,
    cfg: TableRunConfig,
    eval_set: list[str],
    thresholds: list[float],
    stat: str,
    out_prefix: str,
    denominators: dict[str, dict[str, float]],
    bootstrap_samples: int | None,
) -> tuple[pl.DataFrame, list[dict[str, Any]], str]:
    args = biostat_cli.RunArgs(
        resources_json=cfg.resources_json,
        table_name=cfg.table_name,
        eval_level="variant",
        stat=stat,
        eval_set=",".join(eval_set),
        filters="none",
        thresholds=",".join(f"{v:g}" for v in thresholds),
        case_total=None,
        ctrl_total=None,
        case_total_by_eval=_format_totals_arg(denominators, "case_total"),
        ctrl_total_by_eval=_format_totals_arg(denominators, "ctrl_total"),
        bootstrap_samples=bootstrap_samples,
        out_fname=out_prefix,
        write_missing="none",
    )
    out_df, eval_filter_timings, _ = biostat_cli.run(args)
    out_tsv = f"{out_prefix}.tsv"
    log_json = f"{out_prefix}_log.json"
    write_tsv(out_df, out_tsv)
    write_json(
        {
            "run_args": asdict(args),
            "table_path": get_table_config(load_resources(cfg.resources_json), cfg.table_name).path,
            "output_files": {"tsv": out_tsv, "log": log_json},
            "elapsed_seconds": sum(float(t["elapsed_seconds"]) for t in eval_filter_timings),
            "eval_filter_elapsed_seconds": eval_filter_timings,
        },
        log_json,
    )
    return out_df, eval_filter_timings, out_tsv


def _method_label(method: str, display_names: dict[str, str]) -> str:
    return display_names.get(method, method)


def _build_panel_table(
    *,
    raw_df: pl.DataFrame | None,
    pairwise_df: pl.DataFrame | None,
    config: PipelineConfig,
    threshold: float,
    eval_set: list[str],
    panel_eval_map: dict[str, str],
    panel_titles: dict[str, str],
) -> pl.DataFrame:
    panel_by_eval = {eval_name: panel_id for panel_id, eval_name in panel_eval_map.items()}
    method_rank = {name: idx for idx, name in enumerate(config.method_order)}

    frames: list[pl.DataFrame] = []
    for metric_family, df in [("raw", raw_df), ("pairwise", pairwise_df)]:
        if df is None or df.is_empty():
            continue
        expr = (
            pl.col("eval_name").is_in(eval_set)
            & (pl.col("filter_name") == config.default_filter_name)
            & (pl.col("threshold") == threshold)
        )
        subset = df.filter(expr)
        if subset.is_empty():
            continue
        subset = subset.with_columns(
            [
                pl.col("eval_name")
                .map_elements(lambda v: panel_by_eval.get(str(v), "NA"), return_dtype=pl.String)
                .alias("panel_id"),
                pl.col("eval_name")
                .map_elements(lambda v: panel_titles.get(panel_by_eval.get(str(v), ""), str(v)), return_dtype=pl.String)
                .alias("panel_title"),
                pl.lit(metric_family).alias("metric_family"),
                (pl.col("rows_used") / pl.col("total_eval_rows")).alias("rows_used_frac"),
                pl.col("score_name")
                .map_elements(lambda v: _method_label(str(v), config.method_display_names), return_dtype=pl.String)
                .alias("method_label"),
                pl.col("score_name")
                .map_elements(lambda v: method_rank.get(str(v), 10_000), return_dtype=pl.Int64)
                .alias("method_rank"),
            ]
        )
        frames.append(subset)

    if not frames:
        return pl.DataFrame()

    panel_df = pl.concat(frames, how="diagonal_relaxed").sort(
        ["metric_family", "panel_id", "stat", "method_rank", "score_name"]
    )
    return panel_df


def _build_qc_summary(
    panel_df: pl.DataFrame,
    config: PipelineConfig,
    mode: str,
    panel_order: list[str],
    panel_metrics: dict[str, dict[str, str]],
) -> tuple[pl.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    families = ["raw", "pairwise"] if mode == "both" else [mode]

    for family in families:
        fam_df = panel_df.filter(pl.col("metric_family") == family)
        for panel_id in panel_order:
            panel_stat = panel_metrics[panel_id][family]
            count = (
                fam_df.filter((pl.col("panel_id") == panel_id) & (pl.col("stat") == panel_stat))
                .height
            )
            rows.append(
                {
                    "check": "panel_stat_row_count",
                    "metric_family": family,
                    "panel_id": panel_id,
                    "stat": panel_stat,
                    "value": float(count),
                    "status": "ok" if count > 0 else "warning",
                }
            )
            if count == 0:
                warnings.append(f"No rows for panel `{panel_id}` with stat `{panel_stat}` in `{family}` mode.")

    if mode in {"pairwise", "both"} and not panel_df.is_empty():
        pw = panel_df.filter(pl.col("metric_family") == "pairwise")
        if "adjustment_ratio" in pw.columns:
            anchor_rows = pw.filter(
                pl.col("score_name").is_in(["mpc_score", "mpc_score_percentile", "mpc_score_anchor_percentile"])
            )
            if anchor_rows.height > 0:
                deviations = (
                    anchor_rows.with_columns((pl.col("adjustment_ratio") - 1.0).abs().alias("abs_dev"))
                    .select(pl.col("abs_dev").max())
                    .item()
                )
                status = "ok" if deviations <= 1e-6 else "warning"
                rows.append(
                    {
                        "check": "pairwise_anchor_adjustment_ratio_max_abs_deviation",
                        "metric_family": "pairwise",
                        "panel_id": "all",
                        "stat": "pairwise_*",
                        "value": float(deviations),
                        "status": status,
                    }
                )
                if status != "ok":
                    warnings.append("Pairwise anchor adjustment ratio is not consistently ~1.")

    qc_df = pl.DataFrame(rows) if rows else pl.DataFrame(schema={"check": pl.String, "status": pl.String})
    return qc_df, warnings


def _format_method_tick(method_label: str, rows_used_frac: float | None) -> str:
    if rows_used_frac is None or (isinstance(rows_used_frac, float) and math.isnan(rows_used_frac)):
        return method_label
    return f"{method_label}\n({rows_used_frac * 100:.1f}%)"


def _render_mode_figure(
    panel_df: pl.DataFrame,
    *,
    mode: str,
    config: PipelineConfig,
    out_png: str,
    out_pdf: str,
    panel_order: list[str],
    panel_titles: dict[str, str],
    panel_metrics: dict[str, dict[str, str]],
) -> None:
    df = panel_df.filter(pl.col("metric_family") == mode)
    n_panels = len(panel_order)
    n_cols = 2
    n_rows = max(2, (n_panels + 1) // 2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.5 * n_rows), constrained_layout=True)
    flat_axes = list(axes.ravel()) if n_rows > 1 else [axes[0], axes[1]] if n_cols > 1 else [axes]
    for idx, panel_id in enumerate(panel_order):
        if idx >= len(flat_axes):
            break
        ax = flat_axes[idx]
        panel_stat = panel_metrics[panel_id][mode]
        sub = df.filter((pl.col("panel_id") == panel_id) & (pl.col("stat") == panel_stat)).sort(
            ["method_rank", "score_name"]
        )
        if sub.is_empty():
            ax.set_title(f"{panel_id}: {panel_titles.get(panel_id, panel_id)}")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_xticks([])
            continue

        methods = sub["method_label"].to_list()
        values = sub["value"].to_list()
        stderrs = sub["std_error"].to_list()
        rows_used_frac = sub["rows_used_frac"].to_list()
        x = list(range(len(methods)))
        yerr = [float("nan") if (v is None or (isinstance(v, float) and math.isnan(v))) else v for v in stderrs]
        ax.errorbar(x, values, yerr=yerr, fmt="o", capsize=3)
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([_format_method_tick(m, r) for m, r in zip(methods, rows_used_frac)], rotation=35, ha="right")
        ax.set_ylabel(panel_stat)
        ax.set_title(f"{panel_id}: {panel_titles.get(panel_id, panel_id)}")

    for idx in range(n_panels, len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.suptitle(f"Figure 1-style panels ({mode})")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)


def _render_combined_figure(
    panel_df: pl.DataFrame,
    config: PipelineConfig,
    out_png: str,
    out_pdf: str,
    panel_order: list[str],
    panel_titles: dict[str, str],
    panel_metrics: dict[str, dict[str, str]],
) -> None:
    families = [f for f in ["raw", "pairwise"] if f in panel_df["metric_family"].unique().to_list()]
    n_cols = len(families)
    n_rows = len(panel_order)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * max(n_cols, 1), 2.7 * n_rows), squeeze=False, constrained_layout=True)

    for r, panel_id in enumerate(panel_order):
        for c, family in enumerate(families):
            ax = axes[r][c]
            panel_stat = panel_metrics[panel_id][family]
            sub = panel_df.filter(
                (pl.col("metric_family") == family) & (pl.col("panel_id") == panel_id) & (pl.col("stat") == panel_stat)
            ).sort(["method_rank", "score_name"])
            if sub.is_empty():
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_xticks([])
                ax.set_title(f"{panel_id} {family}")
                continue
            methods = sub["method_label"].to_list()
            values = sub["value"].to_list()
            x = list(range(len(methods)))
            ax.scatter(x, values)
            ax.axhline(1.0, linestyle="--", linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=35, ha="right")
            ax.set_ylabel(panel_stat)
            ax.set_title(f"{panel_id}: {panel_titles.get(panel_id, panel_id)} [{family}]")

    fig.suptitle("Figure 1-style panels (combined)")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)


def _render_plots(
    panel_df: pl.DataFrame,
    config: PipelineConfig,
    outdir: str,
    mode: str,
    output_layout: str,
    panel_order: list[str],
    panel_titles: dict[str, str],
    panel_metrics: dict[str, dict[str, str]],
) -> list[str]:
    outputs: list[str] = []

    def render_combined_outputs() -> list[str]:
        combined_outputs: list[str] = []
        if mode in {"raw", "both"}:
            raw_png = str(Path(outdir) / "figure1_raw.png")
            raw_pdf = str(Path(outdir) / "figure1_raw.pdf")
            _render_mode_figure(
                panel_df, mode="raw", config=config, out_png=raw_png, out_pdf=raw_pdf,
                panel_order=panel_order, panel_titles=panel_titles, panel_metrics=panel_metrics,
            )
            combined_outputs.extend([raw_png, raw_pdf])
        if mode in {"pairwise", "both"}:
            pw_png = str(Path(outdir) / "figure1_pairwise.png")
            pw_pdf = str(Path(outdir) / "figure1_pairwise.pdf")
            _render_mode_figure(
                panel_df, mode="pairwise", config=config, out_png=pw_png, out_pdf=pw_pdf,
                panel_order=panel_order, panel_titles=panel_titles, panel_metrics=panel_metrics,
            )
            combined_outputs.extend([pw_png, pw_pdf])
        if mode == "both":
            cmb_png = str(Path(outdir) / "figure1_combined.png")
            cmb_pdf = str(Path(outdir) / "figure1_combined.pdf")
            _render_combined_figure(
                panel_df, config=config, out_png=cmb_png, out_pdf=cmb_pdf,
                panel_order=panel_order, panel_titles=panel_titles, panel_metrics=panel_metrics,
            )
            combined_outputs.extend([cmb_png, cmb_pdf])
        return combined_outputs

    def render_per_eval_outputs() -> list[str]:
        per_eval_outputs: list[str] = []
        per_eval_dir = Path(outdir) / "per_eval"
        per_eval_dir.mkdir(parents=True, exist_ok=True)
        for panel_id in panel_order:
            single_panel_order = [panel_id]
            single_panel_titles = {panel_id: panel_titles.get(panel_id, panel_id)}
            single_panel_metrics = {panel_id: panel_metrics.get(panel_id, {"raw": "enrichment", "pairwise": "pairwise_enrichment"})}
            single_panel_df = panel_df.filter(pl.col("panel_id") == panel_id)
            if single_panel_df.is_empty():
                continue
            safe_panel_id = panel_id.replace("/", "_").replace(" ", "_")
            if mode in {"raw", "both"}:
                raw_png = str(per_eval_dir / f"figure1_raw_{safe_panel_id}.png")
                raw_pdf = str(per_eval_dir / f"figure1_raw_{safe_panel_id}.pdf")
                _render_mode_figure(
                    single_panel_df, mode="raw", config=config, out_png=raw_png, out_pdf=raw_pdf,
                    panel_order=single_panel_order, panel_titles=single_panel_titles, panel_metrics=single_panel_metrics,
                )
                per_eval_outputs.extend([raw_png, raw_pdf])
            if mode in {"pairwise", "both"}:
                pw_png = str(per_eval_dir / f"figure1_pairwise_{safe_panel_id}.png")
                pw_pdf = str(per_eval_dir / f"figure1_pairwise_{safe_panel_id}.pdf")
                _render_mode_figure(
                    single_panel_df, mode="pairwise", config=config, out_png=pw_png, out_pdf=pw_pdf,
                    panel_order=single_panel_order, panel_titles=single_panel_titles, panel_metrics=single_panel_metrics,
                )
                per_eval_outputs.extend([pw_png, pw_pdf])
        return per_eval_outputs

    if output_layout in {"combined", "both"}:
        outputs.extend(render_combined_outputs())
    if output_layout in {"per_eval", "both"}:
        outputs.extend(render_per_eval_outputs())

    return outputs


def _resolve_panel_df_for_plot(
    args: PipelineArgs,
    config: PipelineConfig,
    outdir: str,
    eval_set: list[str] | None = None,
    panel_eval_map: dict[str, str] | None = None,
    panel_titles: dict[str, str] | None = None,
) -> tuple[pl.DataFrame, str]:
    if args.panel_table:
        return pl.read_csv(args.panel_table, separator="\t"), args.panel_table

    raw_df = pl.read_csv(args.raw_metrics_tsv, separator="\t") if args.raw_metrics_tsv else None
    pairwise_df = pl.read_csv(args.pairwise_metrics_tsv, separator="\t") if args.pairwise_metrics_tsv else None
    if raw_df is None and pairwise_df is None:
        raise ValueError(
            "plot requires one of: --panel-table OR --raw-metrics-tsv/--pairwise-metrics-tsv."
        )
    threshold_for_panels = args.threshold if args.threshold is not None else config.default_threshold
    panel_df = _build_panel_table(
        raw_df=raw_df,
        pairwise_df=pairwise_df,
        config=config,
        threshold=threshold_for_panels,
        eval_set=eval_set or config.eval_set,
        panel_eval_map=panel_eval_map or config.panel_eval_map,
        panel_titles=panel_titles or config.panel_titles,
    )
    panel_path = str(Path(outdir) / "panel_table.tsv")
    write_tsv(panel_df, panel_path)
    return panel_df, panel_path


def _write_qc_report(
    qc_summary: pl.DataFrame,
    warnings: list[str],
    out_path: str,
    *,
    config_path: str,
    outdir: str,
) -> None:
    lines = [
        "# Figure 1 Pipeline QC Report",
        "",
        f"- config: `{config_path}`",
        f"- outdir: `{outdir}`",
        "",
        "## Warnings",
    ]
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    lines.extend(["", "## Checks", ""])
    if qc_summary.is_empty():
        lines.append("No checks were produced.")
    else:
        for row in qc_summary.to_dicts():
            lines.append(
                f"- [{row.get('status', 'unknown')}] {row.get('check')} | "
                f"family={row.get('metric_family')} panel={row.get('panel_id')} "
                f"stat={row.get('stat')} value={row.get('value')}"
            )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def _execute_compute(
    args: PipelineArgs,
    config: PipelineConfig,
    outdir: str,
    cfg_path: str,
    eval_set: list[str],
    panel_eval_map: dict[str, str],
    panel_titles: dict[str, str],
    panel_metrics: dict[str, dict[str, str]],
) -> dict[str, Any]:
    panel_order = list(panel_eval_map.keys())

    errors, warnings = _validate_config(
        config,
        args.mode,
        raw_parquet=args.raw_parquet,
        pairwise_parquet=args.pairwise_parquet,
        eval_set=eval_set,
        panel_eval_map=panel_eval_map,
        panel_titles=panel_titles,
        panel_metrics=panel_metrics,
        paper_strict=args.paper_strict,
    )
    if errors:
        raise ValueError("Preflight validation failed:\n- " + "\n- ".join(errors))

    resources_path, raw_run_cfg, pairwise_run_cfg = _materialize_run_resources(
        outdir,
        args.mode,
        raw_parquet=args.raw_parquet,
        pairwise_parquet=args.pairwise_parquet,
        raw_score_columns=config.raw_score_columns,
        pairwise_score_columns=config.pairwise_score_columns,
    )

    threshold_for_panels = args.threshold if args.threshold is not None else config.default_threshold
    thresholds_for_compute = _parse_thresholds(args.thresholds) or config.thresholds
    raw_df: pl.DataFrame | None = None
    pairwise_df: pl.DataFrame | None = None
    metric_outputs: dict[str, Any] = {}

    def run_combined_compute() -> tuple[pl.DataFrame | None, pl.DataFrame | None, dict[str, str]]:
        nonlocal raw_df, pairwise_df
        combined_outputs: dict[str, str] = {}
        if args.mode in {"raw", "both"}:
            assert raw_run_cfg is not None
            raw_prefix = str(Path(outdir) / "metrics_raw")
            raw_df, _timings, raw_path = _run_biostat(
                cfg=raw_run_cfg,
                eval_set=eval_set,
                thresholds=thresholds_for_compute,
                stat="enrichment,rate_ratio",
                out_prefix=raw_prefix,
                denominators=config.rate_ratio_denominators,
                bootstrap_samples=args.bootstrap,
            )
            combined_outputs["raw_tsv"] = raw_path

        if args.mode in {"pairwise", "both"}:
            assert pairwise_run_cfg is not None
            pw_prefix = str(Path(outdir) / "metrics_pairwise")
            pairwise_df, _timings, pw_path = _run_biostat(
                cfg=pairwise_run_cfg,
                eval_set=eval_set,
                thresholds=thresholds_for_compute,
                stat="pairwise_enrichment,pairwise_rate_ratio",
                out_prefix=pw_prefix,
                denominators=config.rate_ratio_denominators,
                bootstrap_samples=args.bootstrap,
            )
            combined_outputs["pairwise_tsv"] = pw_path
        return raw_df, pairwise_df, combined_outputs

    def run_per_eval_compute() -> dict[str, list[str]]:
        per_eval_outputs: dict[str, list[str]] = {"raw_tsvs": [], "pairwise_tsvs": []}
        per_eval_dir = Path(outdir) / "per_eval"
        per_eval_dir.mkdir(parents=True, exist_ok=True)
        for eval_name in eval_set:
            safe_eval = eval_name.replace("/", "_").replace(" ", "_")
            if args.mode in {"raw", "both"}:
                assert raw_run_cfg is not None
                raw_prefix = str(per_eval_dir / f"metrics_raw_{safe_eval}")
                _, _, raw_path = _run_biostat(
                    cfg=raw_run_cfg,
                    eval_set=[eval_name],
                    thresholds=thresholds_for_compute,
                    stat="enrichment,rate_ratio",
                    out_prefix=raw_prefix,
                    denominators=config.rate_ratio_denominators,
                    bootstrap_samples=args.bootstrap,
                )
                per_eval_outputs["raw_tsvs"].append(raw_path)
            if args.mode in {"pairwise", "both"}:
                assert pairwise_run_cfg is not None
                pw_prefix = str(per_eval_dir / f"metrics_pairwise_{safe_eval}")
                _, _, pw_path = _run_biostat(
                    cfg=pairwise_run_cfg,
                    eval_set=[eval_name],
                    thresholds=thresholds_for_compute,
                    stat="pairwise_enrichment,pairwise_rate_ratio",
                    out_prefix=pw_prefix,
                    denominators=config.rate_ratio_denominators,
                    bootstrap_samples=args.bootstrap,
                )
                per_eval_outputs["pairwise_tsvs"].append(pw_path)
        return per_eval_outputs

    if args.output_layout in {"combined", "both"}:
        raw_df, pairwise_df, combined_metric_outputs = run_combined_compute()
        metric_outputs.update(combined_metric_outputs)
    if args.output_layout in {"per_eval", "both"}:
        per_eval_metric_outputs = run_per_eval_compute()
        metric_outputs["per_eval"] = per_eval_metric_outputs
        if args.output_layout == "per_eval" and raw_df is None and pairwise_df is None:
            all_raw_dfs = []
            all_pw_dfs = []
            per_eval_dir = Path(outdir) / "per_eval"
            for eval_name in eval_set:
                safe_eval = eval_name.replace("/", "_").replace(" ", "_")
                if args.mode in {"raw", "both"}:
                    raw_path = per_eval_dir / f"metrics_raw_{safe_eval}.tsv"
                    if raw_path.exists():
                        all_raw_dfs.append(pl.read_csv(str(raw_path), separator="\t"))
                if args.mode in {"pairwise", "both"}:
                    pw_path = per_eval_dir / f"metrics_pairwise_{safe_eval}.tsv"
                    if pw_path.exists():
                        all_pw_dfs.append(pl.read_csv(str(pw_path), separator="\t"))
            if all_raw_dfs:
                raw_df = pl.concat(all_raw_dfs, how="diagonal_relaxed")
            if all_pw_dfs:
                pairwise_df = pl.concat(all_pw_dfs, how="diagonal_relaxed")

    panel_df = _build_panel_table(
        raw_df=raw_df,
        pairwise_df=pairwise_df,
        config=config,
        threshold=threshold_for_panels,
        eval_set=eval_set,
        panel_eval_map=panel_eval_map,
        panel_titles=panel_titles,
    )
    panel_path = str(Path(outdir) / "panel_table.tsv")
    write_tsv(panel_df, panel_path)

    qc_summary, qc_warnings = _build_qc_summary(panel_df, config, args.mode, panel_order, panel_metrics)
    qc_path = str(Path(outdir) / "qc_summary.tsv")
    write_tsv(qc_summary, qc_path)
    all_warnings = [*warnings, *qc_warnings]

    qc_report_path = str(Path(outdir) / "qc_report.md")
    _write_qc_report(qc_summary, all_warnings, qc_report_path, config_path=cfg_path, outdir=outdir)

    return {
        "figure1_run_resources_json": resources_path,
        "raw_parquet": args.raw_parquet,
        "pairwise_parquet": args.pairwise_parquet,
        "panel_table_tsv": panel_path,
        "qc_summary_tsv": qc_path,
        "qc_report_md": qc_report_path,
        "warnings": all_warnings,
        "profile": args.profile,
        "output_layout": args.output_layout,
        "eval_set": eval_set,
        **metric_outputs,
    }


def _enforce_parquet_args(parser: argparse.ArgumentParser, args: PipelineArgs) -> None:
    if args.command not in {"run", "compute"}:
        return
    if args.mode in {"raw", "both"} and not (args.raw_parquet and args.raw_parquet.strip()):
        parser.error("--raw-parquet is required when --mode is raw or both")
    if args.mode in {"pairwise", "both"} and not (args.pairwise_parquet and args.pairwise_parquet.strip()):
        parser.error("--pairwise-parquet is required when --mode is pairwise or both")


def main() -> None:
    parser = _build_parser()
    ns = parser.parse_args()
    args = PipelineArgs(
        command=ns.command,
        config=ns.config,
        mode=ns.mode,
        profile=ns.profile,
        output_layout=getattr(ns, "output_layout", "combined").replace("-", "_"),
        outdir=ns.outdir,
        threshold=ns.threshold,
        thresholds=ns.thresholds,
        bootstrap=ns.bootstrap,
        raw_parquet=getattr(ns, "raw_parquet", None),
        pairwise_parquet=getattr(ns, "pairwise_parquet", None),
        panel_table=getattr(ns, "panel_table", None),
        raw_metrics_tsv=getattr(ns, "raw_metrics_tsv", None),
        pairwise_metrics_tsv=getattr(ns, "pairwise_metrics_tsv", None),
        eval_set_override=getattr(ns, "eval_set", None),
        paper_strict=getattr(ns, "paper_strict", False),
        overwrite=ns.overwrite,
        dry_run=ns.dry_run,
    )
    _enforce_parquet_args(parser, args)

    config, cfg_path = _load_pipeline_config(args.config)
    outdir = args.outdir or _default_outdir(args, cfg_path)

    eval_set, panel_eval_map, panel_titles, panel_metrics, resolve_warnings = _resolve_eval_set(args, config)
    panel_order = list(panel_eval_map.keys())

    if args.command == "plot":
        _prepare_outdir(outdir, overwrite=args.overwrite)
        panel_df, panel_table = _resolve_panel_df_for_plot(
            args, config=config, outdir=outdir,
            eval_set=eval_set, panel_eval_map=panel_eval_map, panel_titles=panel_titles,
        )
        plot_paths = _render_plots(
            panel_df, config=config, outdir=outdir, mode=args.mode, output_layout=args.output_layout,
            panel_order=panel_order, panel_titles=panel_titles, panel_metrics=panel_metrics,
        )
        write_json(
            {
                "command": "plot",
                "mode": args.mode,
                "profile": args.profile,
                "output_layout": args.output_layout,
                "config": cfg_path,
                "panel_table": panel_table,
                "outdir": outdir,
                "plot_outputs": plot_paths,
            },
            str(Path(outdir) / "run_manifest.json"),
        )
        print(f"Plot outputs written to: {outdir}")
        return

    if args.dry_run:
        print("Dry run summary:")
        print(f"  command={args.command}")
        print(f"  mode={args.mode}")
        print(f"  profile={args.profile}")
        print(f"  output_layout={args.output_layout}")
        print(f"  config={cfg_path}")
        print(f"  outdir={outdir}")
        print(f"  eval_set={','.join(eval_set)}")
        print(f"  bootstrap={args.bootstrap}")
        print(f"  paper_strict={args.paper_strict}")
        print(f"  overwrite={args.overwrite}")
        if resolve_warnings:
            print("  resolve warnings:")
            for w in resolve_warnings:
                print(f"    - {w}")
        if args.command in {"run", "compute"}:
            print(f"  raw_parquet={args.raw_parquet}")
            print(f"  pairwise_parquet={args.pairwise_parquet}")
            errs, warns = _validate_config(
                config,
                args.mode,
                raw_parquet=args.raw_parquet,
                pairwise_parquet=args.pairwise_parquet,
                eval_set=eval_set,
                panel_eval_map=panel_eval_map,
                panel_titles=panel_titles,
                panel_metrics=panel_metrics,
                paper_strict=args.paper_strict,
            )
            if warns:
                print("  validation warnings:")
                for w in warns:
                    print(f"    - {w}")
            if errs:
                print("  validation errors:")
                for e in errs:
                    print(f"    - {e}")
                raise SystemExit(2)
        return

    _prepare_outdir(outdir, overwrite=args.overwrite)
    compute_outputs = _execute_compute(
        args, config=config, outdir=outdir, cfg_path=cfg_path,
        eval_set=eval_set, panel_eval_map=panel_eval_map, panel_titles=panel_titles, panel_metrics=panel_metrics,
    )
    plot_outputs: list[str] = []
    if args.command == "run":
        panel_df = pl.read_csv(compute_outputs["panel_table_tsv"], separator="\t")
        plot_outputs = _render_plots(
            panel_df, config=config, outdir=outdir, mode=args.mode, output_layout=args.output_layout,
            panel_order=panel_order, panel_titles=panel_titles, panel_metrics=panel_metrics,
        )

    manifest = {
        "command": args.command,
        "mode": args.mode,
        "profile": args.profile,
        "output_layout": args.output_layout,
        "config": cfg_path,
        "outdir": outdir,
        "outputs": compute_outputs,
        "plot_outputs": plot_outputs,
    }
    write_json(manifest, str(Path(outdir) / "run_manifest.json"))
    print(f"Pipeline outputs written to: {outdir}")


if __name__ == "__main__":
    main()
