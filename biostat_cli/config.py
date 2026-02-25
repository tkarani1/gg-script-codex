from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = [90.0, 95.0, 98.0, 99.0]
ALL_STATS = {"auc", "auprc", "enrichment", "rate_ratio"}


@dataclass(frozen=True)
class TableConfig:
    name: str
    path: str
    level: str
    score_cols: list[str]
    filters: dict[str, str]
    evals: list[str]


def load_resources(resources_json: str) -> dict[str, Any]:
    with Path(resources_json).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_table_config(resources: dict[str, Any], table_name: str) -> TableConfig:
    table_info = resources.get("Table_info", {})
    if table_name not in table_info:
        raise KeyError(f"Unknown table-name: {table_name}")

    item = table_info[table_name]
    return TableConfig(
        name=table_name,
        path=item["Path"],
        level=str(item["Level"]).lower(),
        score_cols=list(item.get("Score_cols", [])),
        filters=dict(item.get("Filters", {})),
        evals=list(item.get("evals", item.get("Evals", []))),
    )


def parse_csv_arg(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_thresholds(raw: str | None) -> list[float]:
    if raw is None or not raw.strip():
        return list(DEFAULT_THRESHOLDS)
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_stats(raw: str) -> set[str]:
    value = raw.strip().lower()
    if value == "all":
        return set(ALL_STATS)
    stats = {part.strip().lower() for part in value.split(",") if part.strip()}
    unknown = stats - ALL_STATS
    if unknown:
        raise ValueError(f"Unknown stat(s): {sorted(unknown)}")
    return stats
