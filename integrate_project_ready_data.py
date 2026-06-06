from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from week_utils import iso_year_week_from_series, year_week_to_monday


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "project_ready_data"

SOURCE_FILES = {
    "historical_videos": ROOT / "youtube_category_output_historical" / "historical_videos_merged.csv",
    "current_videos": ROOT / "youtube_category_output" / "videos.csv",
    "root_videos": ROOT / "videos.csv",
    "team_external_10000plus": ROOT / "team_external_data" / "processed" / "youtube_api_dataset_10000plus_categorized.csv",
    "expanded_recent_20cat": ROOT / "youtube_category_output_expanded_fast" / "expanded_recent_videos_latest.csv",
    "merged_training": ROOT / "youtube_category_output_merged" / "merged_training_dataset.csv",
    "merged_videos": ROOT / "youtube_category_output_merged" / "videos_merged.csv",
    "single_videos": ROOT / "youtube_category_output_single" / "videos.csv",
    "organized_canonical_videos": ROOT / "organized_data" / "canonical" / "videos.csv",
    "project_core_videos": ROOT / "project_core_data" / "videos.csv",
    "final_training_dataset": ROOT / "youtube_category_output_final" / "final_training_dataset.csv",
    "organized_final_training_dataset": ROOT / "organized_data" / "training" / "final_training_dataset.csv",
    "timeseries_support": ROOT / "youtube_category_output" / "tracked_early_timeseries_dataset.csv",
    "timeseries_readiness": ROOT / "youtube_category_output" / "tracked_early_readiness.csv",
    "multimodal_training_tracked": ROOT / "youtube_category_output" / "tracked_early_multimodal_training.csv",
    "multimodal_training_early": ROOT / "youtube_category_output" / "early_multimodal_training.csv",
    "multimodal_training_single": ROOT / "youtube_category_output_single" / "recent_multimodal_training.csv",
    "multimodal_training_final": ROOT / "youtube_category_output_final" / "final_training_dataset.csv",
    "multimodal_training_organized_final": ROOT / "organized_data" / "training" / "final_training_dataset.csv",
}

CATEGORY_ALIAS = {
    "mukbang": "먹방",
    "먹방": "먹방",
    "fitness": "운동",
    "운동": "운동",
    "gaming": "게임",
    "게임": "게임",
    "vlog": "브이로그",
    "브이로그": "브이로그",
    "study": "공부",
    "공부": "공부",
    "finance": "경제",
    "경제": "경제",
    "beauty": "뷰티",
    "뷰티": "뷰티",
    "cooking": "요리",
    "요리": "요리",
    "interiorlife": "인테리어라이프",
    "인테리어라이프": "인테리어라이프",
    "fashion": "패션",
    "패션": "패션",
    "travel": "여행",
    "여행": "여행",
    "tech": "테크",
    "테크": "테크",
    "movie": "영화드라마",
    "drama": "영화드라마",
    "영화드라마": "영화드라마",
    "music": "음악",
    "음악": "음악",
    "pet": "반려동물",
    "반려동물": "반려동물",
    "education": "교육",
    "교육": "교육",
    "news": "뉴스시사",
    "뉴스시사": "뉴스시사",
    "health": "건강",
    "건강": "건강",
    "car": "자동차",
    "자동차": "자동차",
    "parenting": "육아",
    "육아": "육아",
}

VIDEO_KEEP_COLS = [
    "source_dataset",
    "category",
    "source_query",
    "video_id",
    "channel_id",
    "channel_title",
    "title",
    "description",
    "tags",
    "published_at",
    "published_date",
    "year_week",
    "youtube_category_id",
    "category_id",
    "duration",
    "view_count",
    "like_count",
    "comment_count",
    "virality_proxy",
    "current_virality_score",
    "title_length",
    "description_length",
    "tag_count",
    "has_tags",
    "has_shorts_tag",
    "has_hashtag_token",
    "thumbnail_url",
    "collected_at",
]

NUMERIC_COLUMNS = [
    "view_count",
    "like_count",
    "comment_count",
    "final_view_count",
    "final_like_count",
    "final_comment_count",
    "current_virality_score",
    "virality_proxy",
]

TAG_SPLIT_RE = re.compile(r"[|,;/]")
HASHTAG_RE = re.compile(r"#\w+", re.UNICODE)


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding_errors="replace")


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def normalize_category(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    key = re.sub(r"\s+", "", text).lower()
    return CATEGORY_ALIAS.get(key, text)


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    a_num = pd.to_numeric(a, errors="coerce")
    b_num = pd.to_numeric(b, errors="coerce")
    return a_num / b_num.replace(0, np.nan)


def parse_tags(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = [part.strip() for part in TAG_SPLIT_RE.split(text) if part.strip()]
    return parts


def count_hashtags(*values: object) -> int:
    total = 0
    for value in values:
        if pd.isna(value):
            continue
        total += len(HASHTAG_RE.findall(str(value)))
    return total


def normalize_video_dataset(name: str, path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        df = read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    if df.empty or "video_id" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["source_dataset"] = name
    df["category"] = df["category"].map(normalize_category) if "category" in df.columns else pd.NA

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("view_count", "like_count", "comment_count"):
        if col not in df.columns:
            df[col] = pd.NA

    if "final_view_count" in df.columns:
        df["view_count"] = df["view_count"].fillna(df["final_view_count"])
    if "final_like_count" in df.columns:
        df["like_count"] = df["like_count"].fillna(df["final_like_count"])
    if "final_comment_count" in df.columns:
        df["comment_count"] = df["comment_count"].fillna(df["final_comment_count"])

    for col in ("view_count", "like_count", "comment_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    published = pd.to_datetime(df.get("published_at"), errors="coerce", utc=True)
    df["published_at"] = published.astype("string")
    df["published_date"] = published.dt.strftime("%Y-%m-%d")
    df["year_week"] = iso_year_week_from_series(published)

    df["title"] = df.get("title", pd.Series([""] * len(df))).fillna("").astype(str)
    df["description"] = df.get("description", pd.Series([""] * len(df))).fillna("").astype(str)
    df["tags"] = df.get("tags", pd.Series([""] * len(df))).fillna("").astype(str)

    tag_lists = df["tags"].map(parse_tags)
    df["title_length"] = df["title"].str.len()
    df["description_length"] = df["description"].str.len()
    df["tag_count"] = tag_lists.map(len)
    df["has_tags"] = (df["tag_count"] > 0).astype(int)
    df["has_shorts_tag"] = tag_lists.map(
        lambda tags: int(any(tag.lower() in {"shorts", "쇼츠", "#shorts", "#쇼츠"} for tag in tags))
    )
    df["has_hashtag_token"] = [
        count_hashtags(title, desc, tags)
        for title, desc, tags in zip(df["title"], df["description"], df["tags"])
    ]

    if "virality_proxy" not in df.columns or df["virality_proxy"].isna().all():
        df["virality_proxy"] = df["view_count"] + df["like_count"] * 20 + df["comment_count"] * 40

    df = ensure_columns(df, VIDEO_KEEP_COLS)
    return df[VIDEO_KEEP_COLS]


def build_video_pool() -> tuple[pd.DataFrame, pd.DataFrame]:
    video_sources = [
        "historical_videos",
        "current_videos",
        "root_videos",
        "team_external_10000plus",
        "expanded_recent_20cat",
        "merged_training",
        "merged_videos",
        "single_videos",
        "organized_canonical_videos",
        "project_core_videos",
        "final_training_dataset",
        "organized_final_training_dataset",
    ]
    frames = [normalize_video_dataset(name, SOURCE_FILES[name]) for name in video_sources]

    expanded_historical_dir = ROOT / "youtube_category_output_expanded_fast" / "historical"
    if expanded_historical_dir.exists():
        for path in sorted(expanded_historical_dir.glob("historical_videos_offset_*_limit_*.csv")):
            frames.append(normalize_video_dataset("expanded_historical_20cat", path))

    historical_2020_2025 = (
        ROOT / "youtube_category_output_expanded_fast" / "historical_2020_2025" / "historical_2020_2025_latest.csv"
    )
    if historical_2020_2025.exists():
        frames.append(normalize_video_dataset("expanded_historical_2020_2025", historical_2020_2025))

    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    all_rows = pd.concat(frames, ignore_index=True, sort=False)
    all_rows = all_rows[all_rows["video_id"].notna()].copy()

    priority = {
        "expanded_historical_2020_2025": 6,
        "expanded_recent_20cat": 5,
        "expanded_historical_20cat": 4,
        "current_videos": 3,
        "merged_training": 2,
        "historical_videos": 1,
    }
    all_rows["_source_priority"] = all_rows["source_dataset"].map(priority).fillna(0)
    all_rows["_collected_sort"] = pd.to_datetime(all_rows["collected_at"], errors="coerce", utc=True)

    dedup = (
        all_rows.sort_values(["video_id", "_source_priority", "_collected_sort"])
        .drop_duplicates("video_id", keep="last")
        .drop(columns=["_source_priority", "_collected_sort"])
        .reset_index(drop=True)
    )
    all_rows = all_rows.drop(columns=["_source_priority", "_collected_sort"]).reset_index(drop=True)
    return all_rows, dedup


def aggregate_tags(video_pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = video_pool[video_pool["category"].notna() & video_pool["year_week"].notna()].copy()
    for (category, year_week), group in valid.groupby(["category", "year_week"], dropna=False):
        all_tags: list[str] = []
        for value in group["tags"]:
            all_tags.extend(parse_tags(value))
        counter = Counter(tag.lower() for tag in all_tags if tag)
        unique_tags = len(counter)
        top_tags = " | ".join(tag for tag, _ in counter.most_common(5))
        rows.append(
            {
                "category": category,
                "year_week": year_week,
                "unique_tag_count": unique_tags,
                "tag_diversity_score": unique_tags / max(len(group), 1),
                "top_tags": top_tags,
            }
        )
    return pd.DataFrame(rows)


def build_timeseries_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    timeseries = pd.DataFrame()
    readiness = pd.DataFrame()

    ts_path = SOURCE_FILES["timeseries_support"]
    if ts_path.exists() and ts_path.stat().st_size > 0:
        try:
            timeseries = read_csv(ts_path)
        except pd.errors.EmptyDataError:
            timeseries = pd.DataFrame()

    readiness_path = SOURCE_FILES["timeseries_readiness"]
    if readiness_path.exists() and readiness_path.stat().st_size > 0:
        try:
            readiness = read_csv(readiness_path)
        except pd.errors.EmptyDataError:
            readiness = pd.DataFrame()

    if not timeseries.empty:
        if "category" in timeseries.columns:
            timeseries["category"] = timeseries["category"].map(normalize_category)
        for col in timeseries.columns:
            if any(token in col for token in ["views", "likes", "comments", "hours", "count", "score"]):
                timeseries[col] = pd.to_numeric(timeseries[col], errors="coerce")
        published = pd.to_datetime(timeseries.get("published_at"), errors="coerce", utc=True)
        timeseries["published_at"] = published.astype("string")
        timeseries["year_week"] = iso_year_week_from_series(published)
        if {"t1_views", "t3_views"}.issubset(timeseries.columns):
            timeseries["growth_views_1_3"] = timeseries["t3_views"] - timeseries["t1_views"]
        if {"t3_views", "t6_views"}.issubset(timeseries.columns):
            timeseries["growth_views_3_6"] = timeseries["t6_views"] - timeseries["t3_views"]
        if {"t6_views", "t24_views"}.issubset(timeseries.columns):
            timeseries["growth_views_6_24"] = timeseries["t24_views"] - timeseries["t6_views"]
        for prefix in ("t1", "t3", "t6", "t24"):
            v = f"{prefix}_views"
            l = f"{prefix}_likes"
            c = f"{prefix}_comments"
            if {v, l, c}.issubset(timeseries.columns):
                timeseries[f"{prefix}_engagement_rate"] = safe_divide(
                    timeseries[l] + timeseries[c],
                    timeseries[v],
                )

    return timeseries, readiness


def build_timeseries_weekly_features(timeseries: pd.DataFrame) -> pd.DataFrame:
    if timeseries.empty:
        return pd.DataFrame()

    agg_spec = {
        # Count tracked videos per category-week rather than leaking an ID-like column.
        "video_count": ("video_id", "nunique"),
        "latest_hours_since_publish": ("latest_hours_since_publish", "mean"),
        "upload_hour": ("upload_hour", "mean"),
        "upload_hour_sin": ("upload_hour_sin", "mean"),
        "upload_hour_cos": ("upload_hour_cos", "mean"),
        "latest_view_count": ("latest_view_count", "mean"),
        "latest_like_count": ("latest_like_count", "mean"),
        "latest_comment_count": ("latest_comment_count", "mean"),
        "current_virality_score": ("current_virality_score", "mean"),
        "t1_views": ("t1_views", "mean"),
        "t1_likes": ("t1_likes", "mean"),
        "t1_comments": ("t1_comments", "mean"),
        "t3_views": ("t3_views", "mean"),
        "t3_likes": ("t3_likes", "mean"),
        "t3_comments": ("t3_comments", "mean"),
        "t6_views": ("t6_views", "mean"),
        "t6_likes": ("t6_likes", "mean"),
        "t6_comments": ("t6_comments", "mean"),
        "t24_views": ("t24_views", "mean"),
        "t24_likes": ("t24_likes", "mean"),
        "t24_comments": ("t24_comments", "mean"),
        "growth_views_1_3": ("growth_views_1_3", "mean"),
        "growth_views_3_6": ("growth_views_3_6", "mean"),
        "growth_views_6_24": ("growth_views_6_24", "mean"),
        "t1_engagement_rate": ("t1_engagement_rate", "mean"),
        "t3_engagement_rate": ("t3_engagement_rate", "mean"),
        "t6_engagement_rate": ("t6_engagement_rate", "mean"),
        "t24_engagement_rate": ("t24_engagement_rate", "mean"),
    }
    usable = {key: value for key, value in agg_spec.items() if value[0] in timeseries.columns}
    if not usable:
        return pd.DataFrame()

    weekly = (
        timeseries.groupby(["category", "year_week"], dropna=False)
        .agg(**{f"ts_{k}": v for k, v in usable.items()})
        .reset_index()
    )
    weekly["week_date"] = weekly["year_week"].map(year_week_to_monday)
    weekly = weekly.sort_values(["category", "week_date", "year_week"]).reset_index(drop=True)
    if "t24_views" in timeseries.columns:
        readiness = (
            timeseries.assign(ts_has_t24=timeseries["t24_views"].notna().astype(int))
            .groupby(["category", "year_week"], dropna=False)["ts_has_t24"]
            .mean()
            .reset_index()
            .rename(columns={"ts_has_t24": "ts_t24_ready_rate"})
        )
        weekly = weekly.merge(readiness, on=["category", "year_week"], how="left")
    weekly = weekly.drop(columns=["week_date"])
    return weekly


def build_weekly_trend(video_pool: pd.DataFrame, timeseries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if video_pool.empty:
        return pd.DataFrame(), pd.DataFrame()

    valid = video_pool[
        video_pool["category"].notna()
        & video_pool["year_week"].notna()
        & video_pool["video_id"].notna()
    ].copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()

    weekly = (
        valid.groupby(["category", "year_week"], dropna=False)
        .agg(
            video_count=("video_id", "nunique"),
            avg_virality=("virality_proxy", "mean"),
            median_virality=("virality_proxy", "median"),
            avg_views=("view_count", "mean"),
            avg_likes=("like_count", "mean"),
            avg_comments=("comment_count", "mean"),
            unique_channels=("channel_id", "nunique"),
            avg_title_length=("title_length", "mean"),
            avg_description_length=("description_length", "mean"),
            avg_tag_count=("tag_count", "mean"),
            has_tags_rate=("has_tags", "mean"),
            shorts_tag_rate=("has_shorts_tag", "mean"),
            hashtag_signal_rate=("has_hashtag_token", "mean"),
        )
        .reset_index()
    )
    weekly["week_date"] = weekly["year_week"].map(year_week_to_monday)
    weekly = weekly.sort_values(["category", "week_date", "year_week"]).reset_index(drop=True)

    tag_features = aggregate_tags(valid)
    if not tag_features.empty:
        weekly = weekly.merge(tag_features, on=["category", "year_week"], how="left")

    ts_weekly = build_timeseries_weekly_features(timeseries)
    if not ts_weekly.empty:
        weekly = weekly.merge(ts_weekly, on=["category", "year_week"], how="left")
        ts_cols = [col for col in weekly.columns if col.startswith("ts_")]
        if ts_cols:
            weekly = weekly.sort_values(["category", "week_date", "year_week"]).reset_index(drop=True)
            weekly[ts_cols] = (
                weekly.groupby("category", dropna=False)[ts_cols]
                .transform(lambda frame: frame.ffill())
            )

    weekly["lag1_avg_virality"] = weekly.groupby("category")["avg_virality"].shift(1)
    weekly["lag2_avg_virality"] = weekly.groupby("category")["avg_virality"].shift(2)
    weekly["lag1_video_count"] = weekly.groupby("category")["video_count"].shift(1)
    weekly["trend_delta_1"] = weekly["avg_virality"] - weekly["lag1_avg_virality"]
    weekly["trend_delta_2"] = weekly["lag1_avg_virality"] - weekly["lag2_avg_virality"]
    weekly["video_count_delta"] = weekly["video_count"] - weekly["lag1_video_count"]

    weekly["view_per_video"] = safe_divide(weekly["avg_views"], weekly["video_count"])
    weekly["like_per_video"] = safe_divide(weekly["avg_likes"], weekly["video_count"])
    weekly["comment_per_video"] = safe_divide(weekly["avg_comments"], weekly["video_count"])
    weekly["like_view_ratio"] = safe_divide(weekly["avg_likes"], weekly["avg_views"])
    weekly["comment_view_ratio"] = safe_divide(weekly["avg_comments"], weekly["avg_views"])
    weekly["engagement_rate"] = safe_divide(weekly["avg_likes"] + weekly["avg_comments"], weekly["avg_views"])
    weekly["competition_score"] = weekly["video_count"]
    weekly["opportunity_score"] = safe_divide(weekly["avg_virality"], weekly["video_count"])
    weekly["trend_acceleration"] = weekly["trend_delta_1"] - weekly["trend_delta_2"]
    weekly["momentum_ratio"] = safe_divide(weekly["trend_delta_1"], weekly["lag1_avg_virality"])

    weekly["rolling_4week_mean"] = (
        weekly.groupby("category")["avg_virality"].transform(lambda s: s.rolling(window=4, min_periods=1).mean())
    )
    weekly["rolling_4week_std"] = (
        weekly.groupby("category")["avg_virality"].transform(lambda s: s.rolling(window=4, min_periods=2).std())
    )
    weekly["rolling_2week_mean"] = (
        weekly.groupby("category")["avg_virality"].transform(lambda s: s.rolling(window=2, min_periods=1).mean())
    )
    weekly["stability_score"] = safe_divide(weekly["rolling_4week_mean"], weekly["rolling_4week_std"])
    weekly["category_rank"] = weekly.groupby("year_week")["avg_virality"].rank(method="dense", ascending=False)
    weekly["lag1_category_rank"] = weekly.groupby("category")["category_rank"].shift(1)
    weekly["rank_change"] = weekly["lag1_category_rank"] - weekly["category_rank"]

    weekly["creator_entry_score"] = (
        weekly["opportunity_score"].fillna(0) * 0.4
        + weekly["engagement_rate"].fillna(0) * 1000 * 0.25
        + weekly["trend_acceleration"].fillna(0) * 0.05
        + weekly["has_tags_rate"].fillna(0) * 5
        - weekly["competition_score"].fillna(0) * 0.05
    )
    weekly["category_trend_score"] = (
        weekly["rolling_4week_mean"].fillna(0) * 0.5
        + weekly["trend_delta_1"].fillna(0) * 0.3
        + weekly["trend_acceleration"].fillna(0) * 0.2
    )
    weekly["tag_strength_score"] = (
        weekly["avg_tag_count"].fillna(0) * 0.3
        + weekly["tag_diversity_score"].fillna(0) * 0.4
        + weekly["has_tags_rate"].fillna(0) * 10 * 0.3
    )

    if "ts_t1_views" in weekly.columns and "ts_t24_views" in weekly.columns:
        weekly["timeseries_signal_strength"] = safe_divide(
            weekly["ts_t24_views"] - weekly["ts_t1_views"],
            weekly["ts_t1_views"],
        )

    weekly["target_next_avg_virality"] = weekly.groupby("category")["avg_virality"].shift(-1)
    weekly["model_ready"] = weekly["target_next_avg_virality"].notna()

    weekly["iso_year"] = weekly["week_date"].dt.isocalendar().year.astype("Int64")
    weekly["iso_week"] = weekly["week_date"].dt.isocalendar().week.astype("Int64")

    model_table = weekly[weekly["model_ready"]].copy().reset_index(drop=True)
    return weekly.reset_index(drop=True), model_table


def build_multimodal_training_ready() -> pd.DataFrame:
    paths = [
        ("tracked", SOURCE_FILES["multimodal_training_tracked"]),
        ("early", SOURCE_FILES["multimodal_training_early"]),
        ("single", SOURCE_FILES["multimodal_training_single"]),
        ("final", SOURCE_FILES["multimodal_training_final"]),
        ("organized_final", SOURCE_FILES["multimodal_training_organized_final"]),
    ]
    frames: list[pd.DataFrame] = []
    for source_name, path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            df = read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty or "video_id" not in df.columns:
            continue
        df = df.copy()
        df["multimodal_source"] = source_name
        if "category" in df.columns:
            df["category"] = df["category"].map(normalize_category)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged[merged["video_id"].notna()].copy()
    if "published_at" in merged.columns:
        merged["_published_sort"] = pd.to_datetime(merged["published_at"], errors="coerce", utc=True)
    else:
        merged["_published_sort"] = pd.NaT

    source_priority = {
        "final": 5,
        "organized_final": 4,
        "tracked": 3,
        "early": 2,
        "single": 1,
    }
    merged["_source_priority"] = merged["multimodal_source"].map(source_priority).fillna(0)
    merged = (
        merged.sort_values(["video_id", "_source_priority", "_published_sort"])
        .drop_duplicates("video_id", keep="last")
        .drop(columns=["_source_priority", "_published_sort"])
        .reset_index(drop=True)
    )
    return merged


def write_outputs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for name, path in SOURCE_FILES.items():
        manifest_rows.append(
            {
                "dataset": name,
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "last_modified": (
                    datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
                    if path.exists()
                    else None
                ),
            }
        )
    manifest = pd.DataFrame(manifest_rows)

    all_rows, dedup = build_video_pool()
    timeseries, readiness = build_timeseries_outputs()
    weekly, model_table = build_weekly_trend(dedup, timeseries)
    multimodal_ready = build_multimodal_training_ready()

    manifest.to_csv(OUT / "source_manifest.csv", index=False, encoding="utf-8-sig")
    all_rows.to_csv(OUT / "ready_video_pool_all_rows.csv", index=False, encoding="utf-8-sig")
    dedup.to_csv(OUT / "ready_video_pool_dedup.csv", index=False, encoding="utf-8-sig")
    weekly.to_csv(OUT / "ready_category_weekly_trend.csv", index=False, encoding="utf-8-sig")
    model_table.to_csv(OUT / "ready_main_model_table.csv", index=False, encoding="utf-8-sig")
    timeseries.to_csv(OUT / "ready_timeseries_support.csv", index=False, encoding="utf-8-sig")
    readiness.to_csv(OUT / "ready_timeseries_readiness.csv", index=False, encoding="utf-8-sig")
    multimodal_ready.to_csv(OUT / "ready_multimodal_training.csv", index=False, encoding="utf-8-sig")

    if not dedup.empty:
        category_summary = (
            dedup.groupby("category", dropna=False)
            .agg(
                videos=("video_id", "nunique"),
                sources=("source_dataset", lambda s: " | ".join(sorted(set(s.dropna().astype(str))))),
                first_week=("year_week", "min"),
                last_week=("year_week", "max"),
                avg_virality_proxy=("virality_proxy", "mean"),
                avg_views=("view_count", "mean"),
                avg_tag_count=("tag_count", "mean"),
            )
            .reset_index()
            .sort_values("videos", ascending=False)
        )
    else:
        category_summary = pd.DataFrame()
    category_summary.to_csv(OUT / "ready_category_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "video_pool_all_rows": int(len(all_rows)),
        "video_pool_dedup_rows": int(len(dedup)),
        "category_count": int(dedup["category"].nunique()) if not dedup.empty else 0,
        "weekly_trend_rows": int(len(weekly)),
        "main_model_ready_rows": int(len(model_table)),
        "timeseries_rows": int(len(timeseries)),
        "timeseries_video_count": int(timeseries["video_id"].nunique()) if "video_id" in timeseries.columns else 0,
        "multimodal_training_rows": int(len(multimodal_ready)),
        "outputs": {
            "ready_video_pool_all_rows": str(OUT / "ready_video_pool_all_rows.csv"),
            "ready_video_pool_dedup": str(OUT / "ready_video_pool_dedup.csv"),
            "ready_category_weekly_trend": str(OUT / "ready_category_weekly_trend.csv"),
            "ready_main_model_table": str(OUT / "ready_main_model_table.csv"),
            "ready_timeseries_support": str(OUT / "ready_timeseries_support.csv"),
            "ready_multimodal_training": str(OUT / "ready_multimodal_training.csv"),
            "ready_category_summary": str(OUT / "ready_category_summary.csv"),
        },
    }
    (OUT / "ready_data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )

    report = f"""# Project Ready Data Report

Generated at: {summary["generated_at"]}

## Summary

- Video pool all rows: {summary["video_pool_all_rows"]}
- Video pool deduplicated rows: {summary["video_pool_dedup_rows"]}
- Category count: {summary["category_count"]}
- Weekly trend rows: {summary["weekly_trend_rows"]}
- Main model ready rows: {summary["main_model_ready_rows"]}
- Timeseries support rows: {summary["timeseries_rows"]}
- Multimodal training rows: {summary["multimodal_training_rows"]}

## Key Files

- `project_ready_data/ready_video_pool_dedup.csv`
- `project_ready_data/ready_category_weekly_trend.csv`
- `project_ready_data/ready_main_model_table.csv`
- `project_ready_data/ready_timeseries_support.csv`
- `project_ready_data/ready_multimodal_training.csv`

## Notes

Original raw files are preserved. This folder only contains integrated working copies and derived tables for model training.
"""
    (OUT / "READY_DATA_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    write_outputs()
