# BioStat-CLI

Memory-efficient CLI for genomic metric computation from merged Parquet tables.

## Install

```bash
pip install -e .
```

## Run

```bash
biostat-cli \
  --resources-json resources.json \
  --table-name VSM_all_each_per \
  --eval-level variant \
  --stat all \
  --output-tsv out/master.tsv \
  --output-log out/run_params.json
```

## Threshold defaults

- Default thresholds: `90,95,98,99`
- Override with `--thresholds 85,90,99`
