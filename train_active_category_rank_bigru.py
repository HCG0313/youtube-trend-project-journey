from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import holidays
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    holidays = None

try:
    from pytrends.request import TrendReq
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    TrendReq = None

from week_utils import iso_year_week_from_timestamp, year_week_to_monday


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "project_ready_data" / "ready_category_weekly_trend.csv"
OUT_DIR = ROOT / "project_ready_data" / "model_outputs"
EXT_DIR = ROOT / "project_ready_data" / "external_features"

METRICS_CSV_PATH = OUT_DIR / "dl_active_category_rank_bigru_metrics.csv"
SUMMARY_JSON_PATH = OUT_DIR / "dl_active_category_rank_bigru_summary.json"
HOLDOUT_PATH = OUT_DIR / "dl_active_category_rank_bigru_holdout_predictions.csv"
FORECAST_PATH = OUT_DIR / "dl_active_category_rank_bigru_future_probs.csv"
TOP_PATH = OUT_DIR / "dl_active_category_rank_bigru_top_categories.csv"
ACTIVE_PATH = OUT_DIR / "dl_active_category_rank_bigru_active_categories.csv"
SEARCH_CACHE_PATH = EXT_DIR / "google_trends_weekly_active_categories.csv"
SEARCH_NORMALIZED_PATH = EXT_DIR / "google_trends_weekly_active_categories_normalized.csv"
CALENDAR_PATH = EXT_DIR / "korean_weekly_calendar_features.csv"

SEED = 42
SEQ_LEN = 12
HORIZON = 4
ACTIVE_MIN_WEEKS = 5
VAL_START_WEEKS = 4
TEST_START_WEEKS = 4
BATCH_SIZE = 128
WEEKS_PER_BATCH = 4
MAX_EPOCHS = 180
PATIENCE = 24
LR = 8e-4
WEIGHT_DECAY = 7e-5
GRAD_CLIP = 1.0
HIDDEN_SIZE = 96
NUM_LAYERS = 2
DROPOUT = 0.18
EMBED_DIM = 10
FOCAL_GAMMA = 1.5
STEP_LOSS_WEIGHT = 0.15
RANK_LOSS_WEIGHT = 0.15
PAIRWISE_RANK_WEIGHT = 0.40
HORIZON_WEIGHTS = np.array([1.00, 0.95, 0.90, 0.85], dtype=np.float32)
QUANTILE_CANDIDATES = (0.60, 0.65, 0.70)
MIN_RISE_LOG_THRESHOLD = 0.08
THRESHOLD_GRID = np.arange(0.25, 0.751, 0.01)
PAIRWISE_MARGIN = 0.05
MIN_PAIR_GROWTH_DIFF = 0.03
TOP_N = 5
SEARCH_FEATURE_NAMES = ["search_interest", "search_delta_1", "search_roll4", "search_available"]
BLEND_STEP = 0.1

CORE_CATEGORIES = [
    "게임",
    "경제",
    "교육",
    "뉴스시사",
    "먹방",
    "반려동물",
    "뷰티",
    "브이로그",
    "요리",
    "운동",
]

SEARCH_KEYWORDS = {
    "게임": "게임",
    "경제": "경제",
    "공부": "공부",
    "교육": "교육",
    "뉴스시사": "뉴스",
    "먹방": "먹방",
    "반려동물": "반려동물",
    "뷰티": "뷰티",
    "브이로그": "브이로그",
    "여행": "여행",
    "요리": "요리",
    "운동": "운동",
    "음악": "음악",
    "인테리어라이프": "인테리어",
    "자동차": "자동차",
    "테크": "테크",
}

BASE_NUMERIC_COLS = [
    "video_count",
    "avg_virality",
    "engagement_rate",
    "competition_score",
    "opportunity_score",
    "trend_acceleration",
    "momentum_ratio",
    "rolling_4week_mean",
    "rolling_4week_std",
    "category_rank",
    "lag1_avg_virality",
    "lag2_avg_virality",
    "trend_delta_1",
    "trend_delta_2",
    "video_count_delta",
    "creator_entry_score",
    "category_trend_score",
    "tag_strength_score",
    "timeseries_signal_strength",
    "ts_video_count",
    "ts_current_virality_score",
    "ts_growth_views_6_24",
    "ts_t24_ready_rate",
]

SEQUENCE_FEATURE_COLS = [
    "log_avg_virality",
    "log_video_count",
    "observed_flag",
    "activity_flag",
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
    "engagement_rate",
    "competition_score",
    "opportunity_score",
    "trend_acceleration",
    "momentum_ratio",
    "rolling_4week_mean",
    "rolling_4week_std",
    "category_rank",
    "lag1_avg_virality",
    "lag2_avg_virality",
    "trend_delta_1",
    "trend_delta_2",
    "video_count_delta",
    "creator_entry_score",
    "category_trend_score",
    "tag_strength_score",
    "timeseries_signal_strength",
    "ts_video_count",
    "ts_current_virality_score",
    "ts_growth_views_6_24",
    "ts_t24_ready_rate",
    "rel_log_to_roll4",
    "z_to_roll4",
    "category_rank_pct",
    "video_rel_to_roll4",
    "holiday_days_in_week",
    "is_holiday_week",
    "is_long_weekend_week",
    "is_month_start_week",
    "is_month_end_week",
    "is_vacation_period",
    "is_exam_period",
    "search_interest",
    "search_delta_1",
    "search_roll4",
    "search_available",
]

FUTURE_CAL_COLS = [
    "future_holiday_days_sum",
    "future_holiday_week_ratio",
    "future_long_weekend_count",
    "future_month_start_ratio",
    "future_month_end_ratio",
    "future_vacation_ratio",
    "future_exam_ratio",
    "future_month_sin_mean",
    "future_month_cos_mean",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def year_week_to_date(year_week: str) -> pd.Timestamp:
    return year_week_to_monday(year_week)


def date_to_year_week(date: pd.Timestamp) -> str:
    value = iso_year_week_from_timestamp(date)
    if value is None:
        raise ValueError(f"Cannot convert invalid date to year_week: {date}")
    return value


def safe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_recent_activity_counts(raw: pd.DataFrame) -> dict[str, int]:
    work = raw.copy()
    work["week_date"] = work["year_week"].map(year_week_to_date)
    recent_cut = work["week_date"].max() - timedelta(days=7 * 8)
    counts = (
        work.loc[work["week_date"] > recent_cut]
        .groupby("category")["year_week"]
        .nunique()
        .to_dict()
    )
    return {str(k): int(v) for k, v in counts.items()}


def build_calendar_features(week_dates: list[pd.Timestamp]) -> pd.DataFrame:
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    unique_weeks = sorted(pd.Series(week_dates).dropna().unique().tolist())
    if CALENDAR_PATH.exists():
        cached = pd.read_csv(CALENDAR_PATH, parse_dates=["week_date"])
        cached_map = set(cached["week_date"].tolist())
        if len(cached_map) >= len(unique_weeks):
            return cached

    if holidays is None:
        raise ModuleNotFoundError(
            "holidays package is required to build calendar features when no cached calendar file is available."
        )

    years = set()
    for week_date in unique_weeks:
        for offset in range(7):
            years.add((pd.Timestamp(week_date) + timedelta(days=offset)).year)
    kr_holidays = holidays.country_holidays("KR", years=sorted(years))

    rows = []
    for week_date in unique_weeks:
        week_date = pd.Timestamp(week_date)
        days = [week_date + timedelta(days=i) for i in range(7)]
        holiday_days = int(sum(day.date() in kr_holidays for day in days))
        long_weekend = int(
            any((day.date() in kr_holidays) and (day.weekday() in (0, 4)) for day in days) or holiday_days >= 2
        )
        month_start = int(any(day.day <= 7 for day in days))
        month_end = int(any((day.days_in_month - day.day) < 7 for day in days))
        vacation = int(any(day.month in (1, 2, 7, 8) for day in days))
        exam = int(any(day.month in (4, 6, 10, 12) for day in days))
        month = int(week_date.month)

        rows.append(
            {
                "week_date": week_date,
                "holiday_days_in_week": float(holiday_days),
                "is_holiday_week": float(holiday_days > 0),
                "is_long_weekend_week": float(long_weekend),
                "is_month_start_week": float(month_start),
                "is_month_end_week": float(month_end),
                "is_vacation_period": float(vacation),
                "is_exam_period": float(exam),
                "month_sin": float(np.sin(2 * np.pi * month / 12.0)),
                "month_cos": float(np.cos(2 * np.pi * month / 12.0)),
            }
        )

    calendar_df = pd.DataFrame(rows).sort_values("week_date").reset_index(drop=True)
    calendar_df.to_csv(CALENDAR_PATH, index=False, encoding="utf-8-sig")
    return calendar_df


def normalize_search_features(
    raw_search: pd.DataFrame,
    active_categories: list[str],
    min_week: pd.Timestamp,
    max_week: pd.Timestamp,
) -> pd.DataFrame:
    week_index = pd.date_range(start=min_week, end=max_week, freq="W-MON")
    rows = []
    if raw_search.empty:
        for category in active_categories:
            for week_date in week_index:
                rows.append(
                    {
                        "category": category,
                        "week_date": week_date,
                        "search_interest": 0.0,
                        "search_observed": 0.0,
                    }
                )
        return pd.DataFrame(rows)

    work = raw_search.copy()
    work["category"] = work["category"].astype(str)
    work["week_date"] = pd.to_datetime(work["week_date"]).dt.normalize()
    work["week_date"] = work["week_date"] - pd.to_timedelta(work["week_date"].dt.weekday, unit="D")
    work["search_interest"] = pd.to_numeric(work["search_interest"], errors="coerce")
    work = (
        work.groupby(["category", "week_date"], as_index=False)["search_interest"]
        .mean()
        .sort_values(["category", "week_date"])
        .reset_index(drop=True)
    )

    for category in active_categories:
        sub = work.loc[work["category"] == category, ["week_date", "search_interest"]].copy()
        observed = float(not sub.empty)
        if sub.empty:
            for week_date in week_index:
                rows.append(
                    {
                        "category": category,
                        "week_date": week_date,
                        "search_interest": 0.0,
                        "search_observed": 0.0,
                    }
                )
            continue

        sub = sub.set_index("week_date").reindex(week_index)
        sub["search_interest"] = sub["search_interest"].interpolate(method="time", limit_direction="both")
        sub["search_interest"] = sub["search_interest"].fillna(0.0)
        sub = sub.reset_index().rename(columns={"index": "week_date"})
        sub["category"] = category
        sub["search_observed"] = observed
        rows.extend(sub[["category", "week_date", "search_interest", "search_observed"]].to_dict(orient="records"))

    normalized = pd.DataFrame(rows).sort_values(["category", "week_date"]).reset_index(drop=True)
    normalized.to_csv(SEARCH_NORMALIZED_PATH, index=False, encoding="utf-8-sig")
    return normalized


def try_fetch_search_features(active_categories: list[str], min_week: pd.Timestamp, max_week: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, str]]:
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    status = {
        "mode": "none",
        "message": "search_features_not_attempted",
    }
    if SEARCH_NORMALIZED_PATH.exists():
        cached = pd.read_csv(SEARCH_NORMALIZED_PATH, parse_dates=["week_date"])
        status = {
            "mode": "cache",
            "message": "loaded_cached_normalized_google_trends_features",
        }
        return cached, status
    if SEARCH_CACHE_PATH.exists():
        cached = pd.read_csv(SEARCH_CACHE_PATH, parse_dates=["week_date"])
        status = {
            "mode": "cache",
            "message": "loaded_cached_google_trends_features",
        }
        return normalize_search_features(cached, active_categories, min_week, max_week), status

    rows = []
    timeframe = f"{(min_week - timedelta(days=7)).date()} {max_week.date()}"
    failures = []
    if TrendReq is None:
        status = {
            "mode": "none",
            "message": "pytrends_not_installed",
        }
        return pd.DataFrame(), status

    try:
        pytrends = TrendReq(hl="ko-KR", tz=540)
    except Exception as exc:  # pragma: no cover - environment-dependent
        status = {
            "mode": "none",
            "message": f"pytrends_init_failed:{type(exc).__name__}",
        }
        return pd.DataFrame(), status

    for category in active_categories:
        keyword = SEARCH_KEYWORDS.get(category, category)
        try:
            pytrends.build_payload([keyword], timeframe=timeframe, geo="KR")
            interest = pytrends.interest_over_time()
            if interest.empty:
                failures.append(f"{category}:empty")
                continue
            interest = interest.reset_index().rename(columns={"date": "week_date", keyword: "search_interest"})
            interest["week_date"] = pd.to_datetime(interest["week_date"]).dt.normalize()
            interest["week_date"] = interest["week_date"] - pd.to_timedelta(interest["week_date"].dt.weekday, unit="D")
            interest["category"] = category
            if "isPartial" in interest.columns:
                interest = interest.drop(columns="isPartial")
            rows.append(interest[["category", "week_date", "search_interest"]])
            time.sleep(0.5)
        except Exception as exc:  # pragma: no cover - network-dependent
            exc_name = type(exc).__name__
            failures.append(f"{category}:{exc_name}")
            # Google Trends is rate-limited in this environment. If it starts returning
            # TooManyRequests, fail fast instead of spending minutes retrying categories.
            if "TooManyRequests" in exc_name or "429" in str(exc):
                status = {
                    "mode": "none",
                    "message": "google_trends_rate_limited",
                    "failures": ", ".join(failures[:8]),
                }
                return pd.DataFrame(), status

    if not rows:
        status = {
            "mode": "none",
            "message": "google_trends_fetch_failed",
            "failures": ", ".join(failures[:8]),
        }
        return normalize_search_features(pd.DataFrame(), active_categories, min_week, max_week), status

    search_df = pd.concat(rows, ignore_index=True)
    search_df = (
        search_df.groupby(["category", "week_date"], as_index=False)["search_interest"]
        .mean()
        .sort_values(["category", "week_date"])
        .reset_index(drop=True)
    )
    search_df.to_csv(SEARCH_CACHE_PATH, index=False, encoding="utf-8-sig")
    status = {
        "mode": "live",
        "message": "fetched_google_trends_features",
        "failures": ", ".join(failures[:8]) if failures else "",
    }
    return normalize_search_features(search_df, active_categories, min_week, max_week), status


def build_dense_active_grid(
    raw: pd.DataFrame,
    active_categories: list[str],
    calendar_df: pd.DataFrame,
    search_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], bool]:
    work = raw.copy()
    work = work.loc[work["category"].isin(active_categories)].copy()
    work["week_date"] = work["year_week"].map(year_week_to_date)
    work["category"] = work["category"].astype(str)
    work = safe_numeric(work, BASE_NUMERIC_COLS)

    categories = sorted(active_categories)
    category_to_idx = {category: idx for idx, category in enumerate(categories)}
    all_weeks = sorted(work["week_date"].dropna().unique().tolist())
    full_index = pd.MultiIndex.from_product([categories, all_weeks], names=["category", "week_date"])
    dense = full_index.to_frame(index=False)

    merge_cols = ["category", "week_date", *BASE_NUMERIC_COLS]
    merged = dense.merge(work[merge_cols], on=["category", "week_date"], how="left", indicator=True)
    merged["week_date"] = pd.to_datetime(merged["week_date"])
    merged["observed_flag"] = (merged["_merge"] == "both").astype(np.float32)
    merged = merged.drop(columns="_merge")

    merged["avg_virality"] = pd.to_numeric(merged["avg_virality"], errors="coerce").fillna(0.0)
    merged["video_count"] = pd.to_numeric(merged["video_count"], errors="coerce").fillna(0.0)
    merged["activity_flag"] = (merged["video_count"] > 0).astype(np.float32)
    merged["week_of_year"] = merged["week_date"].dt.isocalendar().week.astype(int)
    merged["week_sin"] = np.sin(2 * np.pi * merged["week_of_year"] / 52.0)
    merged["week_cos"] = np.cos(2 * np.pi * merged["week_of_year"] / 52.0)
    merged["log_avg_virality"] = np.log1p(merged["avg_virality"].clip(lower=0.0))
    merged["log_video_count"] = np.log1p(merged["video_count"].clip(lower=0.0))

    merged = merged.merge(calendar_df, on="week_date", how="left")

    has_search = not search_df.empty
    if has_search:
        merged = merged.merge(search_df, on=["category", "week_date"], how="left")
        if "search_observed" in merged.columns:
            merged["search_available"] = pd.to_numeric(merged["search_observed"], errors="coerce").fillna(0.0)
            merged = merged.drop(columns="search_observed")
        else:
            merged["search_available"] = merged["search_interest"].notna().astype(np.float32)
        merged["search_interest"] = pd.to_numeric(merged["search_interest"], errors="coerce").fillna(0.0)
    else:
        merged["search_available"] = 0.0
        merged["search_interest"] = 0.0

    merged = merged.sort_values(["category", "week_date"]).reset_index(drop=True)
    for category, group_idx in merged.groupby("category").groups.items():
        idx = list(group_idx)
        sub = merged.loc[idx].copy()
        roll_video_4 = sub["video_count"].rolling(4, min_periods=1).mean()
        search_roll4 = sub["search_interest"].rolling(4, min_periods=1).mean()
        search_delta = sub["search_interest"].diff().fillna(0.0)
        rank_pct = 1.0 - ((sub["category_rank"].fillna(sub["category_rank"].max()) - 1.0) / max(len(active_categories) - 1, 1))
        rel_log = sub["log_avg_virality"] - np.log1p(sub["rolling_4week_mean"].clip(lower=0.0))
        z_roll = (sub["avg_virality"] - sub["rolling_4week_mean"]) / (sub["rolling_4week_std"].abs() + 1e-6)
        video_rel = sub["log_video_count"] - np.log1p(roll_video_4.clip(lower=0.0))

        merged.loc[idx, "search_roll4"] = search_roll4.to_numpy(dtype=np.float32)
        merged.loc[idx, "search_delta_1"] = search_delta.to_numpy(dtype=np.float32)
        merged.loc[idx, "category_rank_pct"] = rank_pct.fillna(0.0).to_numpy(dtype=np.float32)
        merged.loc[idx, "rel_log_to_roll4"] = rel_log.fillna(0.0).to_numpy(dtype=np.float32)
        merged.loc[idx, "z_to_roll4"] = z_roll.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        merged.loc[idx, "video_rel_to_roll4"] = video_rel.fillna(0.0).to_numpy(dtype=np.float32)

    merged = safe_numeric(merged, SEQUENCE_FEATURE_COLS)
    for col in SEQUENCE_FEATURE_COLS:
        merged[col] = merged[col].fillna(0.0)

    merged["category_idx"] = merged["category"].map(category_to_idx).astype(int)
    merged["year_week"] = merged["week_date"].map(date_to_year_week)
    merged = merged.sort_values(["category_idx", "week_date"]).reset_index(drop=True)
    return merged, category_to_idx, has_search


@dataclass
class RiseArrays:
    seq_x: np.ndarray
    future_cal_x: np.ndarray
    category_idx: np.ndarray
    main_log_growth: np.ndarray
    rank_up: np.ndarray
    step_up: np.ndarray
    start_week: np.ndarray
    category_name: np.ndarray
    current_mean: np.ndarray
    future_mean: np.ndarray
    sample_weight: np.ndarray
    start_week_id: np.ndarray


def build_rise_windows(dense: pd.DataFrame, seq_len: int, horizon: int) -> RiseArrays:
    seq_list = []
    future_cal_list = []
    cat_idx_list = []
    main_growth_list = []
    rank_up_list = []
    step_up_list = []
    start_week_list = []
    category_name_list = []
    current_mean_list = []
    future_mean_list = []
    sample_weight_list = []
    start_week_ids = {}

    min_week = dense["week_date"].min()
    max_week = dense["week_date"].max()
    total_span = max((max_week - min_week).days / 7.0, 1.0)

    for category, group in dense.groupby("category", sort=False):
        group = group.sort_values("week_date").reset_index(drop=True)
        feat = group[SEQUENCE_FEATURE_COLS].to_numpy(dtype=np.float32)
        virality = group["avg_virality"].to_numpy(dtype=np.float32)
        obs = group["observed_flag"].to_numpy(dtype=np.float32)
        rank = group["category_rank"].to_numpy(dtype=np.float32)
        week_dates = group["week_date"].to_numpy()
        category_idx = int(group["category_idx"].iloc[0])

        for end_idx in range(seq_len, len(group) - horizon + 1):
            cur_vals = virality[max(0, end_idx - 4) : end_idx]
            cur_obs = obs[max(0, end_idx - 4) : end_idx]
            fut_vals = virality[end_idx : end_idx + horizon]
            fut_obs = obs[end_idx : end_idx + horizon]
            fut_rank = rank[end_idx : end_idx + horizon]
            current_rank = rank[end_idx - 1]

            if cur_obs.sum() < 2 or fut_obs.sum() < horizon or np.isnan(current_rank):
                continue

            current_mean = float(cur_vals[cur_obs > 0].mean())
            future_mean = float(fut_vals.mean())
            main_log_growth = float(np.log1p(future_mean) - np.log1p(current_mean))
            step_log_growth = np.log1p(fut_vals) - np.log1p(current_mean)
            step_up = (step_log_growth >= 0.0).astype(np.float32)
            rank_up = float(np.nanmean(fut_rank) < current_rank)

            future_slice = group.iloc[end_idx : end_idx + horizon]
            future_cal = np.array(
                [
                    float(future_slice["holiday_days_in_week"].sum()),
                    float(future_slice["is_holiday_week"].mean()),
                    float(future_slice["is_long_weekend_week"].sum()),
                    float(future_slice["is_month_start_week"].mean()),
                    float(future_slice["is_month_end_week"].mean()),
                    float(future_slice["is_vacation_period"].mean()),
                    float(future_slice["is_exam_period"].mean()),
                    float(future_slice["month_sin"].mean()),
                    float(future_slice["month_cos"].mean()),
                ],
                dtype=np.float32,
            )

            recency = 1.0 + (((pd.Timestamp(week_dates[end_idx]) - min_week).days / 7.0) / total_span)
            sample_weight = float(recency * (1.0 + 0.25 * rank_up))

            seq_list.append(feat[end_idx - seq_len : end_idx])
            future_cal_list.append(future_cal)
            cat_idx_list.append(category_idx)
            main_growth_list.append(main_log_growth)
            rank_up_list.append(rank_up)
            step_up_list.append(step_up)
            start_week_list.append(week_dates[end_idx])
            start_week_key = pd.Timestamp(week_dates[end_idx])
            if start_week_key not in start_week_ids:
                start_week_ids[start_week_key] = len(start_week_ids)
            category_name_list.append(category)
            current_mean_list.append(current_mean)
            future_mean_list.append(future_mean)
            sample_weight_list.append(sample_weight)

    return RiseArrays(
        seq_x=np.asarray(seq_list, dtype=np.float32),
        future_cal_x=np.asarray(future_cal_list, dtype=np.float32),
        category_idx=np.asarray(cat_idx_list, dtype=np.int64),
        main_log_growth=np.asarray(main_growth_list, dtype=np.float32),
        rank_up=np.asarray(rank_up_list, dtype=np.float32),
        step_up=np.asarray(step_up_list, dtype=np.float32),
        start_week=np.asarray(start_week_list),
        category_name=np.asarray(category_name_list),
        current_mean=np.asarray(current_mean_list, dtype=np.float32),
        future_mean=np.asarray(future_mean_list, dtype=np.float32),
        sample_weight=np.asarray(sample_weight_list, dtype=np.float32),
        start_week_id=np.asarray([start_week_ids[pd.Timestamp(v)] for v in start_week_list], dtype=np.int64),
    )


def build_start_week_splits(start_weeks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.array(sorted(pd.Series(start_weeks).astype("datetime64[ns]").unique()))
    if len(unique) <= TEST_START_WEEKS + VAL_START_WEEKS + 16:
        raise ValueError("Not enough weeks for temporal split.")
    test = unique[-TEST_START_WEEKS:]
    val = unique[-(TEST_START_WEEKS + VAL_START_WEEKS) : -TEST_START_WEEKS]
    train = unique[: -(TEST_START_WEEKS + VAL_START_WEEKS)]
    return train, val, test


def indices_for_weeks(start_weeks: np.ndarray, selected: np.ndarray) -> np.ndarray:
    mask = pd.Series(start_weeks).astype("datetime64[ns]").isin(pd.Series(selected).astype("datetime64[ns]")).to_numpy()
    return np.where(mask)[0]


def fit_scaler(seq_x: np.ndarray, future_cal_x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = seq_x[train_idx].reshape(-1, seq_x.shape[-1])
    seq_mean = flat.mean(axis=0)
    seq_std = flat.std(axis=0)
    seq_std = np.where(seq_std < 1e-6, 1.0, seq_std)

    cal = future_cal_x[train_idx]
    cal_mean = cal.mean(axis=0)
    cal_std = cal.std(axis=0)
    cal_std = np.where(cal_std < 1e-6, 1.0, cal_std)
    return (
        seq_mean.astype(np.float32),
        seq_std.astype(np.float32),
        cal_mean.astype(np.float32),
        cal_std.astype(np.float32),
    )


def apply_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def build_main_targets(
    arrays: RiseArrays,
    train_idx: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "category": arrays.category_name,
            "main_log_growth": arrays.main_log_growth,
        }
    )
    train_frame = frame.iloc[train_idx]
    thresholds = (
        train_frame.groupby("category")["main_log_growth"]
        .quantile(quantile)
        .clip(lower=MIN_RISE_LOG_THRESHOLD)
        .rename("rise_threshold")
        .reset_index()
    )
    target_frame = frame.merge(thresholds, on="category", how="left")
    targets = (target_frame["main_log_growth"] >= target_frame["rise_threshold"]).astype(np.float32).to_numpy()
    return targets, thresholds


@dataclass
class DatasetArrays:
    seq_x: np.ndarray
    future_cal_x: np.ndarray
    category_idx: np.ndarray
    rise_target: np.ndarray
    rank_up: np.ndarray
    step_up: np.ndarray
    sample_weight: np.ndarray
    main_log_growth: np.ndarray
    start_week_id: np.ndarray


class RiseDataset(Dataset):
    def __init__(self, arrays: DatasetArrays, indices: np.ndarray) -> None:
        self.seq_x = torch.tensor(arrays.seq_x[indices], dtype=torch.float32)
        self.future_cal_x = torch.tensor(arrays.future_cal_x[indices], dtype=torch.float32)
        self.category_idx = torch.tensor(arrays.category_idx[indices], dtype=torch.long)
        self.rise_target = torch.tensor(arrays.rise_target[indices], dtype=torch.float32)
        self.rank_up = torch.tensor(arrays.rank_up[indices], dtype=torch.float32)
        self.step_up = torch.tensor(arrays.step_up[indices], dtype=torch.float32)
        self.sample_weight = torch.tensor(arrays.sample_weight[indices], dtype=torch.float32)
        self.main_log_growth = torch.tensor(arrays.main_log_growth[indices], dtype=torch.float32)
        self.start_week_id = torch.tensor(arrays.start_week_id[indices], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.seq_x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return (
            self.seq_x[idx],
            self.future_cal_x[idx],
            self.category_idx[idx],
            self.rise_target[idx],
            self.rank_up[idx],
            self.step_up[idx],
            self.sample_weight[idx],
            self.main_log_growth[idx],
            self.start_week_id[idx],
        )


class WeekGroupedBatchSampler:
    def __init__(self, start_week_ids: np.ndarray, weeks_per_batch: int, shuffle: bool) -> None:
        week_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, week_id in enumerate(start_week_ids.tolist()):
            week_to_indices[int(week_id)].append(idx)
        self.groups = [group for _week, group in sorted(week_to_indices.items())]
        self.weeks_per_batch = max(int(weeks_per_batch), 1)
        self.shuffle = shuffle

    def __iter__(self):
        group_order = list(range(len(self.groups)))
        if self.shuffle:
            random.shuffle(group_order)
        for batch_start in range(0, len(group_order), self.weeks_per_batch):
            batch_groups = group_order[batch_start : batch_start + self.weeks_per_batch]
            batch = []
            for group_idx in batch_groups:
                batch.extend(self.groups[group_idx])
            if batch:
                yield batch

    def __len__(self) -> int:
        return int(np.ceil(len(self.groups) / self.weeks_per_batch))


class CategoryConditionedBiGRU(nn.Module):
    def __init__(self, input_dim: int, future_cal_dim: int, category_count: int) -> None:
        super().__init__()
        self.category_embed = nn.Embedding(category_count, EMBED_DIM)
        self.seq_proj = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_SIZE),
            nn.GELU(),
            nn.LayerNorm(HIDDEN_SIZE),
        )
        self.gru = nn.GRU(
            input_size=HIDDEN_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.future_proj = nn.Sequential(
            nn.Linear(future_cal_dim, HIDDEN_SIZE // 2),
            nn.GELU(),
            nn.LayerNorm(HIDDEN_SIZE // 2),
        )
        fused_dim = HIDDEN_SIZE * 4 + (HIDDEN_SIZE // 2) + EMBED_DIM
        self.shared = nn.Sequential(
            nn.Linear(fused_dim, HIDDEN_SIZE * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.LayerNorm(HIDDEN_SIZE * 2),
        )
        self.rise_head = nn.Linear(HIDDEN_SIZE * 2, 1)
        self.rank_head = nn.Linear(HIDDEN_SIZE * 2, 1)
        self.step_head = nn.Linear(HIDDEN_SIZE * 2, HORIZON)
        self.score_head = nn.Linear(HIDDEN_SIZE * 2, 1)

    def forward(
        self,
        seq_x: torch.Tensor,
        future_cal_x: torch.Tensor,
        category_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.seq_proj(seq_x)
        out, hidden = self.gru(x)
        hidden = hidden.view(NUM_LAYERS, 2, seq_x.size(0), HIDDEN_SIZE)
        last_hidden = torch.cat([hidden[-1, 0], hidden[-1, 1]], dim=1)
        mean_pool = out.mean(dim=1)
        category_emb = self.category_embed(category_idx)
        future_emb = self.future_proj(future_cal_x)
        fused = torch.cat([last_hidden, mean_pool, category_emb, future_emb], dim=1)
        rep = self.shared(fused)
        rise = self.rise_head(rep).squeeze(1)
        rank = self.rank_head(rep).squeeze(1)
        step = self.step_head(rep)
        score = self.score_head(rep).squeeze(1)
        return rise, rank, step, score


def compute_binary_class_weights(y: np.ndarray) -> tuple[float, float]:
    pos = max(float(y.sum()), 1.0)
    neg = max(float((1.0 - y).sum()), 1.0)
    total = pos + neg
    return neg / total, pos / total


def compute_step_class_weights(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos_list = []
    neg_list = []
    for h in range(y.shape[1]):
        pos, neg = compute_binary_class_weights(y[:, h])
        pos_list.append(pos)
        neg_list.append(neg)
    return np.asarray(pos_list, dtype=np.float32), np.asarray(neg_list, dtype=np.float32)


def focal_bce_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_w: torch.Tensor,
    neg_w: torch.Tensor,
) -> torch.Tensor:
    probs = torch.sigmoid(logits).clamp(1e-5, 1 - 1e-5)
    focal_pos = (1.0 - probs) ** FOCAL_GAMMA
    focal_neg = probs ** FOCAL_GAMMA
    return -(
        targets * pos_w * focal_pos * torch.log(probs)
        + (1.0 - targets) * neg_w * focal_neg * torch.log(1.0 - probs)
    )


def pairwise_margin_ranking_loss(
    scores: torch.Tensor,
    growth_target: torch.Tensor,
    start_week_id: torch.Tensor,
) -> torch.Tensor:
    total_losses = []
    for week_id in torch.unique(start_week_id):
        idx = torch.where(start_week_id == week_id)[0]
        if idx.numel() < 2:
            continue
        week_scores = scores[idx]
        week_growth = growth_target[idx]
        diff = week_growth.unsqueeze(1) - week_growth.unsqueeze(0)
        pair_mask = torch.triu(diff.abs() >= MIN_PAIR_GROWTH_DIFF, diagonal=1)
        if not torch.any(pair_mask):
            continue
        left, right = torch.where(pair_mask)
        pair_target = torch.sign(diff[left, right])
        valid = pair_target != 0
        if not torch.any(valid):
            continue
        left = left[valid]
        right = right[valid]
        pair_target = pair_target[valid]
        pair_loss = nn.functional.margin_ranking_loss(
            week_scores[left],
            week_scores[right],
            pair_target,
            margin=PAIRWISE_MARGIN,
            reduction="none",
        )
        pair_weight = diff[left, right].abs().detach()
        pair_loss = (pair_loss * pair_weight).sum() / pair_weight.sum().clamp_min(1e-6)
        total_losses.append(pair_loss)
    if not total_losses:
        return scores.new_tensor(0.0)
    return torch.stack(total_losses).mean()


def compute_group_ranking_metrics(
    start_weeks: np.ndarray,
    categories: np.ndarray,
    true_rise: np.ndarray,
    growth_target: np.ndarray,
    score: np.ndarray,
    top_n: int = TOP_N,
) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "start_week": pd.Series(start_weeks).astype("datetime64[ns]"),
            "category": categories,
            "true_rise": true_rise.astype(int),
            "growth_target": growth_target.astype(float),
            "score": score.astype(float),
        }
    )
    precisions = []
    recalls = []
    hit_rates = []
    ndcgs = []
    for _week, group in frame.groupby("start_week", sort=True):
        group = group.sort_values("score", ascending=False).reset_index(drop=True)
        top = group.head(top_n)
        positives = int(group["true_rise"].sum())
        precision = float(top["true_rise"].mean()) if len(top) else 0.0
        recall = float(top["true_rise"].sum() / positives) if positives > 0 else 0.0
        hit = float(top["true_rise"].sum() > 0)

        rel = np.maximum(group["growth_target"].to_numpy(dtype=float), 0.0)
        if rel.max() > 0:
            rel = rel / rel.max()
        top_rel = rel[:top_n]
        discounts = 1.0 / np.log2(np.arange(2, len(top_rel) + 2))
        dcg = float(np.sum(top_rel * discounts))
        ideal_rel = np.sort(rel)[::-1][:top_n]
        ideal_dcg = float(np.sum(ideal_rel * discounts[: len(ideal_rel)]))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        hit_rates.append(hit)
        ndcgs.append(ndcg)

    return {
        f"precision_at_{top_n}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall_at_{top_n}": float(np.mean(recalls)) if recalls else 0.0,
        f"hit_rate_at_{top_n}": float(np.mean(hit_rates)) if hit_rates else 0.0,
        f"ndcg_at_{top_n}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def search_best_score_blend(
    start_weeks: np.ndarray,
    categories: np.ndarray,
    true_rise: np.ndarray,
    growth_target: np.ndarray,
    rise_prob: np.ndarray,
    rank_prob: np.ndarray,
    ranking_score: np.ndarray,
    top_n: int = TOP_N,
) -> tuple[tuple[float, float, float], np.ndarray, dict[str, float]]:
    candidates = []
    grid = np.arange(0.0, 1.0 + 1e-9, BLEND_STEP)
    for w_score in grid:
        for w_rise in grid:
            w_rank = 1.0 - w_score - w_rise
            if w_rank < -1e-9:
                continue
            w_rank = max(w_rank, 0.0)
            if abs((w_score + w_rise + w_rank) - 1.0) > 1e-6:
                continue
            candidates.append((float(w_score), float(w_rise), float(w_rank)))
    if not candidates:
        candidates = [(0.6, 0.25, 0.15)]

    best = None
    best_scores = None
    best_metrics = None
    for weights in candidates:
        w_score, w_rise, w_rank = weights
        score = w_score * ranking_score + w_rise * rise_prob + w_rank * rank_prob
        metrics = compute_group_ranking_metrics(
            start_weeks,
            categories,
            true_rise,
            growth_target,
            score,
            top_n=top_n,
        )
        key = (
            metrics[f"ndcg_at_{top_n}"],
            metrics[f"precision_at_{top_n}"],
            metrics[f"recall_at_{top_n}"],
        )
        if best is None or key > best:
            best = key
            best_scores = score
            best_metrics = metrics
            best_weights = weights
    return best_weights, best_scores, best_metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    main_pos_w: float,
    main_neg_w: float,
    rank_pos_w: float,
    rank_neg_w: float,
    step_pos_w: np.ndarray,
    step_neg_w: np.ndarray,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0

    step_pos = torch.tensor(step_pos_w, dtype=torch.float32, device=device).view(1, -1)
    step_neg = torch.tensor(step_neg_w, dtype=torch.float32, device=device).view(1, -1)
    step_weights = torch.tensor(HORIZON_WEIGHTS, dtype=torch.float32, device=device).view(1, -1)
    main_pos = torch.tensor(main_pos_w, dtype=torch.float32, device=device)
    main_neg = torch.tensor(main_neg_w, dtype=torch.float32, device=device)
    rank_pos = torch.tensor(rank_pos_w, dtype=torch.float32, device=device)
    rank_neg = torch.tensor(rank_neg_w, dtype=torch.float32, device=device)

    for (
        seq_x,
        future_cal_x,
        category_idx,
        rise_target,
        rank_up,
        step_up,
        sample_weight,
        main_log_growth,
        start_week_id,
    ) in loader:
        seq_x = seq_x.to(device)
        future_cal_x = future_cal_x.to(device)
        category_idx = category_idx.to(device)
        rise_target = rise_target.to(device)
        rank_up = rank_up.to(device)
        step_up = step_up.to(device)
        sample_weight = sample_weight.to(device)
        main_log_growth = main_log_growth.to(device)
        start_week_id = start_week_id.to(device)

        with torch.set_grad_enabled(training):
            rise_logits, rank_logits, step_logits, score_logits = model(seq_x, future_cal_x, category_idx)
            main_loss = focal_bce_from_logits(rise_logits, rise_target, main_pos, main_neg)
            rank_loss = focal_bce_from_logits(rank_logits, rank_up, rank_pos, rank_neg)
            step_loss = focal_bce_from_logits(step_logits, step_up, step_pos, step_neg)
            step_loss = (step_loss * step_weights).sum(dim=1) / step_weights.sum(dim=1)
            ranking_loss = pairwise_margin_ranking_loss(score_logits, main_log_growth, start_week_id)

            loss = (
                0.30 * main_loss
                + RANK_LOSS_WEIGHT * rank_loss
                + STEP_LOSS_WEIGHT * step_loss
                + PAIRWISE_RANK_WEIGHT * ranking_loss
            )
            loss = (loss * sample_weight).mean()

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

        total_loss += float(loss.item()) * len(seq_x)
        total_count += len(seq_x)

    return total_loss / max(total_count, 1)


def collect_outputs(
    model: nn.Module,
    dataset: RiseDataset,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    rise_probs, rank_probs, step_probs, score_values = [], [], [], []
    model.eval()
    with torch.no_grad():
        for (
            seq_x,
            future_cal_x,
            category_idx,
            _rise_target,
            _rank_up,
            _step_up,
            _sample_weight,
            _main_log_growth,
            _start_week_id,
        ) in loader:
            rise_logits, rank_logits, step_logits, score_logits = model(
                seq_x.to(device),
                future_cal_x.to(device),
                category_idx.to(device),
            )
            rise_probs.append(torch.sigmoid(rise_logits).cpu().numpy())
            rank_probs.append(torch.sigmoid(rank_logits).cpu().numpy())
            step_probs.append(torch.sigmoid(step_logits).cpu().numpy())
            score_values.append(torch.sigmoid(score_logits).cpu().numpy())
    return (
        np.concatenate(rise_probs),
        np.concatenate(rank_probs),
        np.concatenate(step_probs),
        np.concatenate(score_values),
    )


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    acc = float((y_true == y_pred).mean())
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")
    return {
        "accuracy": acc,
        "balanced_accuracy": float((recall + tnr) * 0.5),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    }


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, float]]:
    best_thr = 0.5
    best_metrics = None
    best_bal = -1.0
    best_f1 = -1.0
    best_auc = -1.0
    for thr in THRESHOLD_GRID:
        metrics = compute_binary_metrics(y_true, y_prob, float(thr))
        auc = metrics["roc_auc"] if not np.isnan(metrics["roc_auc"]) else -1.0
        if (
            metrics["balanced_accuracy"] > best_bal + 1e-9
            or (
                abs(metrics["balanced_accuracy"] - best_bal) <= 1e-9
                and metrics["f1"] > best_f1 + 1e-9
            )
            or (
                abs(metrics["balanced_accuracy"] - best_bal) <= 1e-9
                and abs(metrics["f1"] - best_f1) <= 1e-9
                and auc > best_auc
            )
        ):
            best_thr = float(thr)
            best_metrics = metrics
            best_bal = metrics["balanced_accuracy"]
            best_f1 = metrics["f1"]
            best_auc = auc
    return best_thr, best_metrics or compute_binary_metrics(y_true, y_prob, 0.5)


def build_train_sampler(rise_target: np.ndarray, sample_weight: np.ndarray) -> WeightedRandomSampler:
    pos = max(float(rise_target.sum()), 1.0)
    neg = max(float((1.0 - rise_target).sum()), 1.0)
    pos_factor = neg / pos
    weights = np.where(rise_target > 0.5, pos_factor, 1.0).astype(np.float32)
    weights *= sample_weight.astype(np.float32)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def future_calendar_for_next_4_weeks(calendar_df: pd.DataFrame, latest_week: pd.Timestamp) -> np.ndarray:
    target_weeks = [latest_week + timedelta(days=7 * i) for i in range(1, HORIZON + 1)]
    future_slice = calendar_df.loc[calendar_df["week_date"].isin(target_weeks)].sort_values("week_date")
    if len(future_slice) < HORIZON:
        missing = [week for week in target_weeks if week not in set(future_slice["week_date"].tolist())]
        if missing:
            extra = build_calendar_features(missing)
            future_slice = pd.concat([future_slice, extra], ignore_index=True).sort_values("week_date")
    return np.array(
        [
            float(future_slice["holiday_days_in_week"].sum()),
            float(future_slice["is_holiday_week"].mean()),
            float(future_slice["is_long_weekend_week"].sum()),
            float(future_slice["is_month_start_week"].mean()),
            float(future_slice["is_month_end_week"].mean()),
            float(future_slice["is_vacation_period"].mean()),
            float(future_slice["is_exam_period"].mean()),
            float(future_slice["month_sin"].mean()),
            float(future_slice["month_cos"].mean()),
        ],
        dtype=np.float32,
    )


def main() -> None:
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    recent_activity = build_recent_activity_counts(raw)
    active_categories = sorted([cat for cat, n in recent_activity.items() if n >= ACTIVE_MIN_WEEKS])
    core_categories = [cat for cat in CORE_CATEGORIES if cat in active_categories]

    raw["week_date"] = raw["year_week"].map(year_week_to_date)
    min_week = pd.Timestamp(raw["week_date"].min())
    max_week = pd.Timestamp(raw["week_date"].max())

    calendar_df = build_calendar_features(sorted(raw["week_date"].dropna().unique().tolist()))
    search_df, search_status = try_fetch_search_features(active_categories, min_week, max_week)
    dense, category_to_idx, has_search = build_dense_active_grid(raw, active_categories, calendar_df, search_df)
    arrays = build_rise_windows(dense, SEQ_LEN, HORIZON)

    train_weeks, val_weeks, test_weeks = build_start_week_splits(arrays.start_week)
    train_idx = indices_for_weeks(arrays.start_week, train_weeks)
    val_idx = indices_for_weeks(arrays.start_week, val_weeks)
    test_idx = indices_for_weeks(arrays.start_week, test_weeks)

    seq_mean, seq_std, cal_mean, cal_std = fit_scaler(arrays.seq_x, arrays.future_cal_x, train_idx)
    seq_x_scaled = apply_scaler(arrays.seq_x, seq_mean, seq_std)
    future_cal_scaled = apply_scaler(arrays.future_cal_x, cal_mean, cal_std)

    best_run = None
    candidate_rows = []
    search_col_idx = [SEQUENCE_FEATURE_COLS.index(col) for col in SEARCH_FEATURE_NAMES]
    feature_modes = ["calendar_only", "calendar_search"] if has_search else ["calendar_only"]
    search_feature_eligible = bool(has_search)

    for feature_mode in feature_modes:
        feature_seq_x = seq_x_scaled.copy()
        if feature_mode == "calendar_only":
            feature_seq_x[:, :, search_col_idx] = 0.0

        for quantile in QUANTILE_CANDIDATES:
            rise_target, threshold_frame = build_main_targets(arrays, train_idx, quantile)
            data_arrays = DatasetArrays(
                seq_x=feature_seq_x,
                future_cal_x=future_cal_scaled,
                category_idx=arrays.category_idx,
                rise_target=rise_target,
                rank_up=arrays.rank_up,
                step_up=arrays.step_up,
                sample_weight=arrays.sample_weight,
                main_log_growth=arrays.main_log_growth,
                start_week_id=arrays.start_week_id,
            )

            train_dataset = RiseDataset(data_arrays, train_idx)
            val_dataset = RiseDataset(data_arrays, val_idx)
            test_dataset = RiseDataset(data_arrays, test_idx)

            main_pos_w, main_neg_w = compute_binary_class_weights(rise_target[train_idx])
            rank_pos_w, rank_neg_w = compute_binary_class_weights(arrays.rank_up[train_idx])
            step_pos_w, step_neg_w = compute_step_class_weights(arrays.step_up[train_idx])

            train_loader = DataLoader(
                train_dataset,
                batch_sampler=WeekGroupedBatchSampler(
                    arrays.start_week_id[train_idx],
                    weeks_per_batch=WEEKS_PER_BATCH,
                    shuffle=True,
                ),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_sampler=WeekGroupedBatchSampler(
                    arrays.start_week_id[val_idx],
                    weeks_per_batch=WEEKS_PER_BATCH,
                    shuffle=False,
                ),
            )

            model = CategoryConditionedBiGRU(
                input_dim=len(SEQUENCE_FEATURE_COLS),
                future_cal_dim=len(FUTURE_CAL_COLS),
                category_count=len(category_to_idx),
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            best_state = None
            best_val_loss = float("inf")
            patience_left = PATIENCE

            for _epoch in range(1, MAX_EPOCHS + 1):
                _train_loss = run_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    main_pos_w,
                    main_neg_w,
                    rank_pos_w,
                    rank_neg_w,
                    step_pos_w,
                    step_neg_w,
                )
                val_loss = run_epoch(
                    model,
                    val_loader,
                    None,
                    device,
                    main_pos_w,
                    main_neg_w,
                    rank_pos_w,
                    rank_neg_w,
                    step_pos_w,
                    step_neg_w,
                )

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience_left = PATIENCE
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

            if best_state is None:
                raise RuntimeError("Model failed to produce a checkpoint.")
            model.load_state_dict(best_state)

            val_rise_prob, val_rank_prob, val_step_prob, val_score = collect_outputs(model, val_dataset, device)
            val_threshold, val_metrics = select_threshold(rise_target[val_idx].astype(int), val_rise_prob)
            blend_weights, val_final_score, val_rank_metrics = search_best_score_blend(
                arrays.start_week[val_idx],
                arrays.category_name[val_idx],
                rise_target[val_idx].astype(int),
                arrays.main_log_growth[val_idx],
                val_rise_prob,
                val_rank_prob,
                val_score,
                top_n=TOP_N,
            )
            test_rise_prob, test_rank_prob, test_step_prob, test_score = collect_outputs(model, test_dataset, device)
            test_metrics = compute_binary_metrics(rise_target[test_idx].astype(int), test_rise_prob, val_threshold)
            test_final_score = (
                blend_weights[0] * test_score
                + blend_weights[1] * test_rise_prob
                + blend_weights[2] * test_rank_prob
            )
            test_rank_metrics = compute_group_ranking_metrics(
                arrays.start_week[test_idx],
                arrays.category_name[test_idx],
                rise_target[test_idx].astype(int),
                arrays.main_log_growth[test_idx],
                test_final_score,
                top_n=TOP_N,
            )

            row = {
                "feature_mode": feature_mode,
                "quantile": quantile,
                "val_threshold": val_threshold,
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics["roc_auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
                "val_precision_at_5": val_rank_metrics[f"precision_at_{TOP_N}"],
                "val_ndcg_at_5": val_rank_metrics[f"ndcg_at_{TOP_N}"],
                "test_precision_at_5": test_rank_metrics[f"precision_at_{TOP_N}"],
                "test_ndcg_at_5": test_rank_metrics[f"ndcg_at_{TOP_N}"],
                "train_positive_rate": float(rise_target[train_idx].mean()),
                "val_positive_rate": float(rise_target[val_idx].mean()),
                "test_positive_rate": float(rise_target[test_idx].mean()),
            }
            candidate_rows.append(row)

            rank_aux_test_metrics = compute_binary_metrics(arrays.rank_up[test_idx].astype(int), test_rank_prob, 0.5)
            eligible = feature_mode == "calendar_only" or search_feature_eligible
            if not eligible:
                continue

            if best_run is None:
                choose = True
            else:
                prev = best_run["selection_metrics"]
                curr = (
                    val_rank_metrics[f"precision_at_{TOP_N}"],
                    val_rank_metrics[f"ndcg_at_{TOP_N}"],
                    val_metrics["balanced_accuracy"],
                )
                choose = curr > prev

            if choose:
                best_run = {
                    "feature_mode": feature_mode,
                    "quantile": quantile,
                    "model_state": best_state,
                    "threshold_frame": threshold_frame.copy(),
                    "val_threshold": val_threshold,
                    "selection_metrics": (
                        val_rank_metrics[f"precision_at_{TOP_N}"],
                        val_rank_metrics[f"ndcg_at_{TOP_N}"],
                        val_metrics["balanced_accuracy"],
                    ),
                    "val_metrics": val_metrics,
                    "val_rank_metrics": val_rank_metrics,
                    "blend_weights": blend_weights,
                    "test_metrics": test_metrics,
                    "test_rank_metrics": test_rank_metrics,
                    "rank_aux_test_metrics": rank_aux_test_metrics,
                    "rise_target": rise_target.copy(),
                    "test_rise_prob": test_rise_prob.copy(),
                    "test_rank_prob": test_rank_prob.copy(),
                    "test_step_prob": test_step_prob.copy(),
                    "test_score": test_score.copy(),
                    "test_final_score": test_final_score.copy(),
                    "feature_seq_x": feature_seq_x.copy(),
                }

    if best_run is None:
        raise RuntimeError("No candidate model was selected.")

    pd.DataFrame(candidate_rows).to_csv(METRICS_CSV_PATH, index=False, encoding="utf-8-sig")

    final_model = CategoryConditionedBiGRU(
        input_dim=len(SEQUENCE_FEATURE_COLS),
        future_cal_dim=len(FUTURE_CAL_COLS),
        category_count=len(category_to_idx),
    ).to(device)
    final_model.load_state_dict(best_run["model_state"])

    rise_target = best_run["rise_target"]
    holdout = pd.DataFrame(
        {
            "category": arrays.category_name[test_idx],
            "start_week_date": pd.Series(arrays.start_week[test_idx]).astype("datetime64[ns]"),
            "target_main_rise": rise_target[test_idx].astype(int),
            "predicted_rise_probability": best_run["test_rise_prob"],
            "predicted_rise_label": (best_run["test_rise_prob"] >= best_run["val_threshold"]).astype(int),
            "target_rank_up": arrays.rank_up[test_idx].astype(int),
            "predicted_rank_up_probability": best_run["test_rank_prob"],
            "predicted_ranking_score": best_run["test_score"],
            "predicted_final_score": best_run["test_final_score"],
            "current_4week_mean": arrays.current_mean[test_idx],
            "future_4week_mean": arrays.future_mean[test_idx],
            "main_log_growth": arrays.main_log_growth[test_idx],
            "presentation_core": pd.Series(arrays.category_name[test_idx]).isin(core_categories).astype(int),
        }
    )
    holdout.to_csv(HOLDOUT_PATH, index=False, encoding="utf-8-sig")

    latest_week = dense["week_date"].max()
    future_cal = future_calendar_for_next_4_weeks(calendar_df, latest_week)
    future_cal_scaled = apply_scaler(future_cal[None, :], cal_mean, cal_std)
    future_rows = []
    future_category_scores = []
    w_score, w_rise, w_rank = best_run["blend_weights"]
    for category, idx in category_to_idx.items():
        group = dense.loc[dense["category"] == category].sort_values("week_date").copy()
        seq = group[SEQUENCE_FEATURE_COLS].to_numpy(dtype=np.float32)[-SEQ_LEN:]
        seq_scaled = apply_scaler(seq[None, :, :], seq_mean, seq_std)
        if best_run["feature_mode"] == "calendar_only":
            seq_scaled[:, :, search_col_idx] = 0.0
        seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32, device=device)
        cal_tensor = torch.tensor(future_cal_scaled, dtype=torch.float32, device=device)
        cat_tensor = torch.tensor([idx], dtype=torch.long, device=device)

        with torch.no_grad():
            rise_logit, rank_logit, step_logit, score_logit = final_model(seq_tensor, cal_tensor, cat_tensor)
            rise_prob = float(torch.sigmoid(rise_logit).cpu().numpy()[0])
            rank_prob = float(torch.sigmoid(rank_logit).cpu().numpy()[0])
            step_prob = torch.sigmoid(step_logit).cpu().numpy()[0]
            ranking_score = float(torch.sigmoid(score_logit).cpu().numpy()[0])

        future_rows.append(
            {
                "category": category,
                "rise_probability": rise_prob,
                "rank_up_probability": rank_prob,
                "ranking_score": ranking_score,
                "step1_up_probability": float(step_prob[0]),
                "step2_up_probability": float(step_prob[1]),
                "step3_up_probability": float(step_prob[2]),
                "step4_up_probability": float(step_prob[3]),
                "presentation_core": int(category in core_categories),
            }
        )
        future_category_scores.append(
            {
                "category": category,
                "rise_probability": rise_prob,
                "rank_up_probability": rank_prob,
                "ranking_score": ranking_score,
                "final_score": w_score * ranking_score + w_rise * rise_prob + w_rank * rank_prob,
                "presentation_core": int(category in core_categories),
            }
        )

    future_df = pd.DataFrame(future_rows).sort_values("ranking_score", ascending=False).reset_index(drop=True)
    future_df.to_csv(FORECAST_PATH, index=False, encoding="utf-8-sig")

    top_df = pd.DataFrame(future_category_scores).sort_values("final_score", ascending=False).reset_index(drop=True)
    top_df.to_csv(TOP_PATH, index=False, encoding="utf-8-sig")

    active_df = pd.DataFrame(
        {
            "category": active_categories,
            "recent_active_weeks": [recent_activity[c] for c in active_categories],
            "presentation_core": [int(c in core_categories) for c in active_categories],
        }
    )
    active_df.to_csv(ACTIVE_PATH, index=False, encoding="utf-8-sig")

    summary = {
        "model_name": "CategoryConditionedBiGRU_RankingAwareRiseModel",
        "target_definition": "next_4week_mean_log_growth >= category_train_quantile_threshold with pairwise ranking over weekly category groups",
        "target_quantile_selected": best_run["quantile"],
        "main_probability_threshold": best_run["val_threshold"],
        "sequence_length_weeks": SEQ_LEN,
        "forecast_horizon_weeks": HORIZON,
        "top_n_target": TOP_N,
        "active_category_min_recent_weeks": ACTIVE_MIN_WEEKS,
        "active_categories": active_categories,
        "presentation_core_categories": core_categories,
        "search_feature_status": search_status,
        "used_search_features": bool(has_search),
        "search_live_eligible_for_final_model": bool(has_search and search_status.get("mode") == "live"),
        "selected_feature_mode": best_run["feature_mode"],
        "selected_blend_weights": {
            "ranking_score": best_run["blend_weights"][0],
            "rise_probability": best_run["blend_weights"][1],
            "rank_up_probability": best_run["blend_weights"][2],
        },
        "train_windows": int(len(train_idx)),
        "val_windows": int(len(val_idx)),
        "test_windows": int(len(test_idx)),
        "train_positive_rate": float(rise_target[train_idx].mean()),
        "val_positive_rate": float(rise_target[val_idx].mean()),
        "test_positive_rate": float(rise_target[test_idx].mean()),
        "val_metrics": best_run["val_metrics"],
        "val_ranking_metrics": best_run["val_rank_metrics"],
        "test_metrics": best_run["test_metrics"],
        "test_ranking_metrics": best_run["test_rank_metrics"],
        "rank_aux_test_metrics": best_run["rank_aux_test_metrics"],
        "top_predicted_categories": top_df.head(10).to_dict(orient="records"),
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
