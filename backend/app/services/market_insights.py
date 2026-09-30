"""Market insight helpers backed by OEWS and BLS demo datasets."""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from ..utils.data_loader import DataNotFoundError, load_bls_data, load_oews_data

_logger = logging.getLogger(__name__)

SUPPORTED_METRICS = {
    "average_salary",
    "average_salary_adjusted",
    "hourly_mean",
    "hourly_mean_adjusted",
    "median_salary",
    "total_employment",
}

METRIC_ALIASES = {
    "a_mean": "average_salary",
    "h_mean": "hourly_mean",
    "a_mean_adj": "average_salary_adjusted",
    "h_mean_adj": "hourly_mean_adjusted",
}

METRIC_MAP = {
    "average_salary": "a_mean",
    "hourly_mean": "h_mean",
    "average_salary_adjusted": "a_mean_adj",
    "hourly_mean_adjusted": "h_mean_adj",
    "median_salary": "a_median",
    "total_employment": "tot_emp",
}

CSV_TO_NORMALIZED = {csv: metric for metric, csv in METRIC_MAP.items()}


def top_states_for_occupation(
    occupation: str | None,
    metric: str = "average_salary",
    top_n: int = 5,
) -> List[Dict[str, object]]:
    """Rank states for a given occupation using OEWS sample data."""
    if not occupation:
        _logger.warning("top_states_for_occupation called without occupation")
        return []

    raw_metric = (metric or "average_salary").lower()
    metric_key = METRIC_ALIASES.get(raw_metric, raw_metric)
    csv_column = METRIC_MAP.get(metric_key)
    print("=== MARKET_INSIGHTS DEBUG START ===")
    print("raw_metric =", raw_metric)
    print("metric_key =", metric_key)
    print("csv_column =", csv_column)
    print("=== MARKET_INSIGHTS DEBUG END ===")
    if metric_key not in SUPPORTED_METRICS:
        _logger.warning("Unsupported metric '%s' requested", metric_key)
        return []
    if not csv_column:
        _logger.warning("No CSV column found for metric '%s'", metric_key)
        return []

    try:
        df = load_oews_data()
    except DataNotFoundError as exc:
        _logger.warning("OEWS dataset missing: %s", exc)
        return []
    print("Loaded CSV columns:", df.columns.tolist())

    filtered = df[df["occupation"].str.contains(occupation, case=False, na=False)].copy()
    if filtered.empty:
        _logger.warning("No OEWS rows found for occupation '%s'", occupation)
        return []

    agg_map: Dict[str, str] = {}
    for metric_name, csv_name in METRIC_MAP.items():
        if csv_name in filtered.columns:
            agg_map[csv_name] = "sum" if metric_name == "total_employment" else "mean"
    if csv_column not in agg_map:
        _logger.warning("Metric '%s' unavailable in OEWS dataset", metric_key)
        return []

    if metric_key == "total_employment":
        filtered[csv_column] = pd.to_numeric(filtered[csv_column], errors="coerce")
    stats_series = filtered.get(csv_column)
    if stats_series is not None and not stats_series.dropna().empty:
        series = stats_series.dropna().astype(float)
        _logger.warning(
            "[JobMap] %s — min=%s, max=%s, mean=%s",
            occupation,
            round(series.min(), 3),
            round(series.max(), 3),
            round(series.mean(), 3),
        )

    aggregated = (
        filtered.groupby(["state_name", "state_code"], as_index=False)
        .agg(agg_map)
        .dropna(subset=[csv_column])
        .sort_values(csv_column, ascending=False)
        .head(max(top_n, 1))
    )

    aggregated = aggregated.rename(columns=CSV_TO_NORMALIZED)

    records: List[Dict[str, object]] = []
    for row in aggregated.itertuples():
        record = {
            "state_name": getattr(row, "state_name"),
            "state_code": getattr(row, "state_code"),
            "average_salary": _to_float(getattr(row, "average_salary", None)),
            "average_salary_adjusted": _to_float(getattr(row, "average_salary_adjusted", None)),
            "hourly_mean": _to_float(getattr(row, "hourly_mean", None)),
            "hourly_mean_adjusted": _to_float(getattr(row, "hourly_mean_adjusted", None)),
            "median_salary": _to_float(getattr(row, "median_salary", None)),
            "total_employment": _to_int(getattr(row, "total_employment", None)),
        }
        records.append(record)

    _logger.info("Computed top %s states for %s by %s", len(records), occupation, metric_key)
    return records


def macro_trend(series_id: str, periods: int = 12) -> List[Dict[str, object]]:
    """Return chronological macro trend data for the requested BLS series."""
    if not series_id:
        return []

    key = series_id.strip().lower()
    try:
        df = load_bls_data()
    except DataNotFoundError as exc:
        _logger.warning("BLS dataset missing: %s", exc)
        return []

    filtered = df[df["series_id"].str.lower() == key]
    if filtered.empty:
        _logger.warning("No BLS series found for '%s'", series_id)
        return []

    filtered = filtered.sort_values("date")
    if periods and periods > 0:
        filtered = filtered.tail(periods)

    trend: List[Dict[str, object]] = []
    for row in filtered.itertuples():
        date = getattr(row, "date")
        date_str = date.strftime("%Y-%m") if hasattr(date, "strftime") else str(date)
        value = getattr(row, "value")
        trend.append({"date": date_str, "value": float(value) if pd.notna(value) else None})

    _logger.info("Computed macro trend for %s", series_id)
    return trend


def _to_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None
