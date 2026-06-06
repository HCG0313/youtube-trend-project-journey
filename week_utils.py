from __future__ import annotations

from datetime import datetime

import pandas as pd


def iso_year_week_from_series(dates: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dates, errors="coerce", utc=True)
    iso = parsed.dt.isocalendar()
    year = iso["year"].astype("Int64")
    week = iso["week"].astype("Int64")
    result = year.astype("string") + "-" + week.astype("string").str.zfill(2)
    return result.where(parsed.notna(), pd.NA)


def iso_year_week_from_timestamp(value: pd.Timestamp | str | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    iso = ts.isocalendar()
    return f"{iso.year}-{iso.week:02d}"


def year_week_to_monday(year_week: str | object) -> pd.Timestamp:
    if pd.isna(year_week):
        return pd.NaT
    text = str(year_week).strip()
    try:
        year_str, week_str = text.split("-")
        year = int(year_str)
        week = int(week_str)
    except Exception:
        return pd.NaT
    if week <= 0:
        return pd.NaT
    max_week = datetime(year, 12, 28).isocalendar().week
    if week > max_week:
        return pd.NaT
    return pd.Timestamp(datetime.fromisocalendar(year, week, 1))


def iso_week_sort_key(series: pd.Series) -> pd.Series:
    parts = series.astype(str).str.split("-", n=1, expand=True)
    year = pd.to_numeric(parts[0], errors="coerce").fillna(-1).astype(int)
    week = pd.to_numeric(parts[1], errors="coerce").fillna(-1).astype(int)
    return year * 100 + week
