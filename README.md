# BioStat-CLI

Memory-efficient CLI for genomic statistics on merged Parquet tables using Polars lazy execution.

## Features

- Computes `auc`, `auprc`, `enrichment`, and `rate_ratio`
- Supports `variant` and `gene` eval levels (strategy-based evaluators)
- Reads local paths and `gs://` parquet inputs via Polars/fsspec/gcsfs
- Writes:
  - main metrics TSV
  - run log JSON (args, resolved table path, total runtime, per eval/filter runtime)
- Optional missing-variant TSV to explain `rows_used` vs `total_eval_rows`
- Includes a parallel runner (`biostat_cli.cli_parallel`) for eval/filter concurrency

## Install

```bash
cd /Users/tk508/Work/new/gg-script-codex
pip install -e .
```

## Resources JSON format

`--resources-json` defaults to `resources.json`.

```json
{
  "Table_info": {
    "VSM_all_inner_per_v1": {
      "Path": "gs://bucket/path/to/data.parquet",
      "Level": "variant",
      "Score_cols": ["AM_percentile", "score_PAI3D_percentile"],
      "Filters": {"ordered": "filter_ordered"},
      "evals": ["is_pos__schema", "is_pos_dd"]
    }
  }
}
```

## CLI arguments

- `--resources-json` path to resources file (default: `resources.json`)
- `--table-name` table key under `Table_info` (**required**)
- `--eval-level` `variant` or `gene` (**required**)
- `--stat` `all` or csv subset (`auc,auprc,enrichment,rate_ratio`)
- `--eval-set` optional csv eval override (defaults to all `evals` from resources)
- `--filters` optional csv logical filter names (from `Filters` keys); `none` is always included
- `--thresholds` optional csv thresholds
- `--case-total`, `--ctrl-total` optional denominators for rate ratio
- `--out-fname` output naming schema/prefix (**required**)
- `--write-missing` controls missing-entity report: `none`, `all`, or `any` (default: `none`)

Output paths are derived from `--out-fname`:

- main TSV: `<schema>.tsv`
- log JSON: `<schema>_log.json`
- missing TSV: `<schema>_missing.tsv` (when `--write-missing` is `all` or `any`)

## Threshold behavior

- Thresholds are percentile-based **fractions in `[0,1]`**
- Default thresholds: `0.90,0.95,0.98,0.99`
- Passing any threshold `> 1.0` exits with error code `22`

Example:

```bash
--thresholds 0.90,0.95,0.98,0.99
```

## Main output TSV schema

Columns:

- `eval_name`
- `filter_name`
- `score_name`
- `threshold`
- `stat`
- `value`
- `p_value`
- `tp`, `fp`, `tn`, `fn`
- `rows_used`
- `total_eval_rows`

## Missing-output TSV (optional)

Use to inspect score-null entities after eval/filter masking.

- One row per entity (per eval/filter)
- `all` mode: only variants missing score values in all methods
- `any` mode: variants missing in one or more methods, with category for all vs partial
- Missing report sort order:
  - first by `eval_name`
  - then by `filter_name` (when present)
  - then by `missing_category` (`all_methods` before `partial_methods`)
  - then variant-level chromosome/position (`chr1`..`chr22`, `chrX`, `chrY`) or gene identifier/name

Columns:

- `eval_name`
- `filter_name`
- id columns are auto-detected for both levels:
  - variant: `chrom,pos,ref,alt` or `locus,alleles`
  - gene: `GENE_ID`, `gene_id`, `ensg`, or `gene_symbol`
- `missing_category` (`all_methods` or `partial_methods`)
- `missing_score_count`
- `missing_score_names`

## Runtime logging

The generated log JSON (`<schema>_log.json`) includes:

- `run_args`
- `table_path`
- `output_files`
- `elapsed_seconds`
- `eval_filter_elapsed_seconds` (per eval/filter runtime)

## Examples

### Standard runner

```bash
python -m biostat_cli.cli \
  --resources-json ../files/vsm_all.json \
  --table-name VSM_all_inner_per_v1 \
  --eval-level variant \
  --stat "enrichment,auc" \
  --thresholds 0.90,0.95,0.98,0.99 \
  --out-fname ../results/VSM_v1 \
  --write-missing any
```

### Parallel runner

```bash
python -m biostat_cli.cli_parallel \
  --resources-json ../files/vsm_all.json \
  --table-name VSM_all_inner_per_v1 \
  --eval-level variant \
  --stat "enrichment,auc" \
  --out-fname ../results/VSM_v1_parallel
```
