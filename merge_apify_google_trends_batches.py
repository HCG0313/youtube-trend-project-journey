from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
EXT_DIR = ROOT / "project_ready_data" / "external_features"
TREND_PATH = ROOT / "project_ready_data" / "ready_category_weekly_trend.csv"

INPUTS = [
    EXT_DIR / "google_trends_weekly_active_categories_apify_raw.csv",
    EXT_DIR / "google_trends_apify_priority_recent_batch2_raw.csv",
    EXT_DIR / "google_trends_apify_priority_recent_raw.csv",
]

OUT_RAW = EXT_DIR / "google_trends_apify_final_raw.csv"
OUT_STITCHED = EXT_DIR / "google_trends_apify_final_stitched.csv"
OUT_NORMALIZED = EXT_DIR / "google_trends_apify_final_normalized.csv"
OUT_SUMMARY = EXT_DIR / "google_trends_apify_final_summary.json"


def year_week_to_date(year_week: str) -> pd.Timestamp:
    year_str, week_str = str(year_week).split("-")
    year = int(year_str)
    week = int(week_str)
    if week <= 0:
        return pd.NaT
    max_week = datetime(year, 12, 28).isocalendar().week
    week = min(week, max_week)
    return pd.Timestamp(datetime.fromisocalendar(year, week, 1))


def stitch_all(raw_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for category, sub in raw_df.groupby("category", sort=True):
        chunks = []
        for _, chunk in sub.groupby(["query_start", "query_end", "chunk_index"], sort=True):
            chunks.append(
                chunk.sort_values("week_date")[["category", "week_date", "search_interest_raw"]]
                .rename(columns={"search_interest_raw": "search_interest"})
                .reset_index(drop=True)
            )
        if not chunks:
            continue
        stitched = chunks[0]
        for chunk in chunks[1:]:
            overlap = stitched.merge(chunk, on=["category", "week_date"], how="inner", suffixes=("_old", "_new"))
            valid = overlap[(overlap["search_interest_old"] > 0) & (overlap["search_interest_new"] > 0)]
            scale = float((valid["search_interest_old"] / valid["search_interest_new"]).median()) if not valid.empty else 1.0
            scale = max(0.05, min(20.0, scale))
            work = chunk.copy()
            work["search_interest_scaled"] = work["search_interest"] * scale
            merged = stitched.merge(work[["category", "week_date", "search_interest_scaled"]], on=["category", "week_date"], how="outer")
            both = merged["search_interest"].notna() & merged["search_interest_scaled"].notna()
            merged.loc[both, "search_interest"] = (merged.loc[both, "search_interest"] + merged.loc[both, "search_interest_scaled"]) / 2.0
            merged["search_interest"] = merged["search_interest"].fillna(merged["search_interest_scaled"])
            stitched = merged[["category", "week_date", "search_interest"]].sort_values("week_date").reset_index(drop=True)
        max_value = float(stitched["search_interest"].max()) if not stitched.empty else 0.0
        if max_value > 0:
            stitched["search_interest"] = (stitched["search_interest"] / max_value) * 100.0
        parts.append(stitched)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["category", "week_date", "search_interest"])


def normalize_search_features(stitched: pd.DataFrame) -> pd.DataFrame:
    trend = pd.read_csv(TREND_PATH)
    trend["week_date"] = trend["year_week"].map(year_week_to_date)
    trend = trend.dropna(subset=["category", "week_date"]).copy()
    categories = sorted(trend["category"].dropna().unique().tolist())
    min_week = pd.Timestamp(trend["week_date"].min()).normalize()
    max_week = pd.Timestamp(trend["week_date"].max()).normalize()
    week_index = pd.date_range(start=min_week, end=max_week, freq="W-MON")

    rows = []
    stitched = stitched.copy()
    if not stitched.empty:
        stitched["week_date"] = pd.to_datetime(stitched["week_date"]).dt.normalize()
    for category in categories:
        sub = stitched.loc[stitched["category"] == category, ["week_date", "search_interest"]].copy()
        observed_weeks = set(sub["week_date"].tolist())
        sub = sub.set_index("week_date").reindex(week_index)
        sub["search_interest"] = sub["search_interest"].interpolate(method="time", limit_direction="both")
        sub["search_interest"] = sub["search_interest"].fillna(0.0)
        sub = sub.reset_index().rename(columns={"index": "week_date"})
        sub["category"] = category
        sub["search_observed"] = sub["week_date"].isin(observed_weeks).astype(float)
        rows.extend(sub[["category", "week_date", "search_interest", "search_observed"]].to_dict(orient="records"))
    return pd.DataFrame(rows).sort_values(["category", "week_date"]).reset_index(drop=True)


def main() -> None:
    frames = []
    used_files = []
    for path in INPUTS:
        if path.exists():
            df = pd.read_csv(path, parse_dates=["week_date"])
            if not df.empty:
                frames.append(df)
                used_files.append(path.name)
    if not frames:
        raise SystemExit("No input batch files found.")

    raw = pd.concat(frames, ignore_index=True)
    raw["week_date"] = pd.to_datetime(raw["week_date"]).dt.normalize()
    raw = raw.drop_duplicates(
        subset=["category", "keyword", "chunk_index", "query_start", "query_end", "week_date"],
        keep="last",
    ).sort_values(["category", "query_start", "week_date"]).reset_index(drop=True)
    raw.to_csv(OUT_RAW, index=False, encoding="utf-8-sig")

    stitched = stitch_all(raw)
    stitched.to_csv(OUT_STITCHED, index=False, encoding="utf-8-sig")

    normalized = normalize_search_features(stitched)
    normalized.to_csv(OUT_NORMALIZED, index=False, encoding="utf-8-sig")

    summary = {
        "used_files": used_files,
        "raw_rows": int(len(raw)),
        "stitched_rows": int(len(stitched)),
        "normalized_rows": int(len(normalized)),
        "stitched_weeks_by_category": stitched.groupby("category")["week_date"].nunique().sort_values(ascending=False).to_dict() if not stitched.empty else {},
        "observed_rate_by_category": normalized.groupby("category")["search_observed"].mean().sort_values(ascending=False).to_dict() if not normalized.empty else {},
    }
    with OUT_SUMMARY.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
