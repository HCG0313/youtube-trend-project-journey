from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "project_ready_data" / "ready_category_weekly_trend.csv"
EXT_DIR = ROOT / "project_ready_data" / "external_features"

RAW_OUT_PATH = EXT_DIR / "google_trends_weekly_active_categories_apify_raw.csv"
STITCHED_OUT_PATH = EXT_DIR / "google_trends_weekly_active_categories_apify_stitched.csv"
NORMALIZED_OUT_PATH = EXT_DIR / "google_trends_weekly_active_categories_apify_normalized.csv"
LOG_OUT_PATH = EXT_DIR / "google_trends_apify_collection_log.csv"
SUMMARY_OUT_PATH = EXT_DIR / "google_trends_apify_summary.json"

DEFAULT_NORMALIZED_PATH = EXT_DIR / "google_trends_weekly_active_categories_normalized.csv"
DEFAULT_RAW_PATH = EXT_DIR / "google_trends_weekly_active_categories.csv"

ACTOR_SLUG = "parseforge~google-trends-scraper"
RUN_SYNC_ENDPOINT = f"https://api.apify.com/v2/acts/{ACTOR_SLUG}/run-sync-get-dataset-items"

ACTIVE_MIN_WEEKS = 5
SPAN_WEEKS = 52
OVERLAP_WEEKS = 8
REQUEST_TIMEOUT = 240
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 4.0


CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "게임": ["게임"],
    "경제": ["경제"],
    "공부": ["공부", "스터디"],
    "교육": ["교육", "건강", "입시"],
    "뉴스시사": ["뉴스", "시사"],
    "먹방": ["먹방", "맛집"],
    "반려동물": ["반려동물", "강아지", "고양이"],
    "뷰티": ["뷰티", "메이크업", "화장"],
    "브이로그": ["브이로그", "vlog"],
    "여행": ["여행"],
    "요리": ["요리", "레시피"],
    "운동": ["운동", "헬스", "다이어트"],
    "음악": ["음악", "노래"],
    "인테리어라이프": ["인테리어", "집꾸미기", "라이프스타일"],
    "자동차": ["자동차", "차"],
    "테크": ["테크", "IT", "기술"],
}


@dataclass
class ChunkJob:
    category: str
    chunk_index: int
    monday_start: pd.Timestamp
    monday_end: pd.Timestamp
    query_start: str
    query_end: str
    expected_weeks: int


@dataclass
class ChunkResult:
    category: str
    chunk_index: int
    query_start: str
    query_end: str
    selected_keyword: str
    keyword_candidates: str
    status: str
    points: int
    expected_weeks: int
    apify_error: str
    request_error: str
    attempts: int
    scale_factor: float
    overlap_weeks: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recollect active-category weekly Google Trends via Apify with chunk stitching."
    )
    parser.add_argument("--token", type=str, default="", help="Apify token. If omitted, APIFY_TOKEN is used.")
    parser.add_argument("--token-env", type=str, default="APIFY_TOKEN", help="Environment variable name for the Apify token.")
    parser.add_argument("--active-min-weeks", type=int, default=ACTIVE_MIN_WEEKS)
    parser.add_argument("--span-weeks", type=int, default=SPAN_WEEKS)
    parser.add_argument("--overlap-weeks", type=int, default=OVERLAP_WEEKS)
    parser.add_argument("--max-categories", type=int, default=0, help="Limit number of active categories for smoke tests.")
    parser.add_argument("--categories", type=str, default="", help="Comma-separated category override list.")
    parser.add_argument("--promote-to-default", action="store_true", help="Overwrite the default Google Trends cache files after success.")
    return parser.parse_args()


def year_week_to_date(year_week: str) -> pd.Timestamp:
    year_str, week_str = str(year_week).split("-")
    year = int(year_str)
    week = int(week_str)
    if week <= 0:
        return pd.NaT
    max_week = datetime(year, 12, 28).isocalendar().week
    week = min(week, max_week)
    return pd.Timestamp(datetime.fromisocalendar(year, week, 1))


def load_active_categories(active_min_weeks: int) -> tuple[pd.DataFrame, list[str], pd.Timestamp, pd.Timestamp]:
    raw = pd.read_csv(DATA_PATH)
    raw["week_date"] = raw["year_week"].map(year_week_to_date)
    raw = raw.dropna(subset=["category", "week_date"]).copy()
    recent_cut = raw["week_date"].max() - timedelta(days=7 * 8)
    counts = (
        raw.loc[raw["week_date"] > recent_cut]
        .groupby("category")["year_week"]
        .nunique()
        .sort_values(ascending=False)
    )
    active_categories = counts[counts >= active_min_weeks].index.tolist()
    min_week = pd.Timestamp(raw["week_date"].min()).normalize()
    max_week = pd.Timestamp(raw["week_date"].max()).normalize()
    return raw, active_categories, min_week, max_week


def build_chunk_jobs(
    categories: list[str],
    min_week: pd.Timestamp,
    max_week: pd.Timestamp,
    span_weeks: int,
    overlap_weeks: int,
) -> list[ChunkJob]:
    jobs: list[ChunkJob] = []
    current_start = min_week
    chunk_index = 0
    while current_start <= max_week:
        current_end = min(current_start + timedelta(weeks=span_weeks - 1), max_week)
        query_start = (current_start - timedelta(days=1)).date().isoformat()
        query_end = (current_end + timedelta(days=5)).date().isoformat()
        expected_weeks = max(1, ((current_end - current_start).days // 7) + 1)
        for category in categories:
            jobs.append(
                ChunkJob(
                    category=category,
                    chunk_index=chunk_index,
                    monday_start=current_start,
                    monday_end=current_end,
                    query_start=query_start,
                    query_end=query_end,
                    expected_weeks=expected_weeks,
                )
            )
        if current_end >= max_week:
            break
        current_start = current_end - timedelta(weeks=overlap_weeks - 1)
        current_start += timedelta(weeks=1)
        chunk_index += 1
        if current_start <= min_week and chunk_index > 1:
            raise RuntimeError("Chunk plan did not advance. Check overlap/span settings.")
    return jobs


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "accept": "application/json",
            "content-type": "application/json",
            "user-agent": "codex-google-trends-recollector/1.0",
        }
    )
    return session


def call_actor(
    session: requests.Session,
    token: str,
    keyword: str,
    query_start: str,
    query_end: str,
) -> tuple[dict, str]:
    params = {"token": token, "format": "json", "clean": "true"}
    payload = {"keywords": [keyword], "timeRange": f"{query_start} {query_end}", "geo": "KR"}
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.post(RUN_SYNC_ENDPOINT, params=params, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            items = response.json()
            if not isinstance(items, list) or not items:
                last_error = "empty_response_items"
            else:
                return items[0], ""
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = f"{type(exc).__name__}:{exc}"
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_SLEEP_SECONDS * attempt)
    return {}, last_error


def parse_time_label_to_monday(label: str) -> pd.Timestamp:
    text = (
        str(label)
        .replace("\u2009", " ")
        .replace("\u202f", " ")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .strip()
    )
    if "-" not in text:
        raise ValueError(f"Unexpected time label: {label}")
    left, right = [part.strip() for part in text.split("-", 1)]
    if "," not in left and "," in right:
        year = right.split(",")[-1].strip()
        left = f"{left}, {year}"
    start_date = pd.Timestamp(pd.to_datetime(left, errors="raise"))
    if start_date.weekday() == 6:
        monday = start_date + timedelta(days=1)
    else:
        monday = start_date - timedelta(days=start_date.weekday())
    return monday.normalize()


def parse_interest_frame(item: dict, category: str, keyword: str, chunk_index: int, query_start: str, query_end: str) -> pd.DataFrame:
    rows = item.get("interestOverTime") or []
    parsed = []
    for row in rows:
        try:
            week_date = parse_time_label_to_monday(row.get("time", ""))
            value = float(row.get("value", 0.0))
            parsed.append(
                {
                    "category": category,
                    "keyword": keyword,
                    "chunk_index": chunk_index,
                    "query_start": query_start,
                    "query_end": query_end,
                    "week_date": week_date,
                    "search_interest_raw": value,
                }
            )
        except Exception:
            continue
    return pd.DataFrame(parsed)


def choose_best_keyword_result(
    session: requests.Session,
    token: str,
    job: ChunkJob,
    keyword_candidates: list[str],
) -> tuple[pd.DataFrame, ChunkResult]:
    best_frame = pd.DataFrame()
    best_result = ChunkResult(
        category=job.category,
        chunk_index=job.chunk_index,
        query_start=job.query_start,
        query_end=job.query_end,
        selected_keyword="",
        keyword_candidates=" | ".join(keyword_candidates),
        status="failed",
        points=0,
        expected_weeks=job.expected_weeks,
        apify_error="",
        request_error="",
        attempts=0,
        scale_factor=1.0,
        overlap_weeks=0,
    )

    for keyword in keyword_candidates:
        item, request_error = call_actor(session, token, keyword, job.query_start, job.query_end)
        best_result.attempts += 1
        if request_error:
            best_result.request_error = request_error
            continue

        frame = parse_interest_frame(item, job.category, keyword, job.chunk_index, job.query_start, job.query_end)
        apify_error = str(item.get("error") or "")
        points = len(frame)

        better = False
        if points > len(best_frame):
            better = True
        elif points == len(best_frame) and len(best_frame) > 0 and not apify_error and best_result.apify_error:
            better = True

        if better:
            best_frame = frame
            best_result.selected_keyword = keyword
            best_result.points = points
            best_result.apify_error = apify_error
            best_result.status = "success" if points > 0 else "failed"

        if points >= max(2, math.floor(job.expected_weeks * 0.6)) and not apify_error:
            break

        time.sleep(1.0)

    return best_frame, best_result


def stitch_category_chunks(category_frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, list[tuple[int, float, int]]]:
    if not category_frames:
        return pd.DataFrame(columns=["category", "week_date", "search_interest"]), []

    category_frames = [frame.sort_values("week_date").reset_index(drop=True) for frame in category_frames if not frame.empty]
    if not category_frames:
        return pd.DataFrame(columns=["category", "week_date", "search_interest"]), []

    stitched = category_frames[0][["category", "week_date", "search_interest_raw"]].rename(columns={"search_interest_raw": "search_interest"})
    scale_log: list[tuple[int, float, int]] = []

    for frame in category_frames[1:]:
        work = frame[["category", "week_date", "search_interest_raw"]].copy()
        overlap = stitched.merge(work, on=["category", "week_date"], how="inner")
        valid = overlap[(overlap["search_interest"] > 0) & (overlap["search_interest_raw"] > 0)].copy()
        if not valid.empty:
            ratios = valid["search_interest"] / valid["search_interest_raw"]
            scale_factor = float(np_clip(ratios.median(), 0.05, 20.0))
            overlap_weeks = int(len(valid))
        else:
            scale_factor = 1.0
            overlap_weeks = 0
        scale_log.append((int(frame["chunk_index"].iloc[0]), scale_factor, overlap_weeks))

        work["search_interest_scaled"] = work["search_interest_raw"] * scale_factor
        merged = stitched.merge(
            work[["category", "week_date", "search_interest_scaled"]],
            on=["category", "week_date"],
            how="outer",
        )

        both = merged["search_interest"].notna() & merged["search_interest_scaled"].notna()
        merged.loc[both, "search_interest"] = (
            merged.loc[both, "search_interest"] + merged.loc[both, "search_interest_scaled"]
        ) / 2.0
        merged["search_interest"] = merged["search_interest"].fillna(merged["search_interest_scaled"])
        stitched = merged[["category", "week_date", "search_interest"]].sort_values("week_date").reset_index(drop=True)

    max_value = float(stitched["search_interest"].max()) if not stitched.empty else 0.0
    if max_value > 0:
        stitched["search_interest"] = (stitched["search_interest"] / max_value) * 100.0
    return stitched, scale_log


def np_clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def normalize_search_features(
    stitched: pd.DataFrame,
    categories: list[str],
    min_week: pd.Timestamp,
    max_week: pd.Timestamp,
) -> pd.DataFrame:
    week_index = pd.date_range(start=min_week, end=max_week, freq="W-MON")
    rows: list[dict] = []

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

    normalized = pd.DataFrame(rows).sort_values(["category", "week_date"]).reset_index(drop=True)
    return normalized


def save_outputs(
    raw_df: pd.DataFrame,
    stitched_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    log_df: pd.DataFrame,
    summary: dict,
    promote_to_default: bool,
) -> None:
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(RAW_OUT_PATH, index=False, encoding="utf-8-sig")
    stitched_df.to_csv(STITCHED_OUT_PATH, index=False, encoding="utf-8-sig")
    normalized_df.to_csv(NORMALIZED_OUT_PATH, index=False, encoding="utf-8-sig")
    log_df.to_csv(LOG_OUT_PATH, index=False, encoding="utf-8-sig")
    with SUMMARY_OUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    if promote_to_default:
        raw_df[["category", "week_date", "search_interest"]].to_csv(DEFAULT_RAW_PATH, index=False, encoding="utf-8-sig")
        normalized_df.to_csv(DEFAULT_NORMALIZED_PATH, index=False, encoding="utf-8-sig")


def load_existing_state() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_df = (
        pd.read_csv(RAW_OUT_PATH, parse_dates=["week_date"])
        if RAW_OUT_PATH.exists()
        else pd.DataFrame(columns=["category", "keyword", "chunk_index", "query_start", "query_end", "week_date", "search_interest_raw"])
    )
    stitched_df = (
        pd.read_csv(STITCHED_OUT_PATH, parse_dates=["week_date"])
        if STITCHED_OUT_PATH.exists()
        else pd.DataFrame(columns=["category", "week_date", "search_interest"])
    )
    log_df = (
        pd.read_csv(LOG_OUT_PATH)
        if LOG_OUT_PATH.exists()
        else pd.DataFrame(
            columns=[
                "category",
                "chunk_index",
                "query_start",
                "query_end",
                "selected_keyword",
                "keyword_candidates",
                "status",
                "points",
                "expected_weeks",
                "apify_error",
                "request_error",
                "attempts",
                "scale_factor",
                "overlap_weeks",
            ]
        )
    )
    return raw_df, stitched_df, log_df


def build_summary(
    active_min_weeks: int,
    categories: list[str],
    missing_mapping: list[str],
    min_week: pd.Timestamp,
    max_week: pd.Timestamp,
    span_weeks: int,
    overlap_weeks: int,
    raw_df: pd.DataFrame,
    stitched_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    log_df: pd.DataFrame,
    promote_to_default: bool,
) -> dict:
    observed_rate = (
        normalized_df.groupby("category", observed=False)["search_observed"].mean().sort_values(ascending=False).to_dict()
        if not normalized_df.empty
        else {}
    )
    exact_coverage = float(normalized_df["search_observed"].mean()) if not normalized_df.empty else 0.0
    success_jobs = int((log_df["status"] == "success").sum()) if not log_df.empty else 0
    failed_jobs = int((log_df["status"] != "success").sum()) if not log_df.empty else 0
    completed_categories = []
    if not log_df.empty:
        expected_per_category = int(log_df.groupby("category")["chunk_index"].nunique().max())
        counts = log_df.groupby("category")["chunk_index"].nunique()
        completed_categories = sorted(counts[counts >= expected_per_category].index.tolist())

    return {
        "actor_slug": ACTOR_SLUG,
        "active_min_weeks": int(active_min_weeks),
        "categories_requested": categories,
        "categories_completed": completed_categories,
        "missing_keyword_mapping": missing_mapping,
        "min_week": min_week.date().isoformat(),
        "max_week": max_week.date().isoformat(),
        "span_weeks": int(span_weeks),
        "overlap_weeks": int(overlap_weeks),
        "jobs_total": int(len(log_df)),
        "jobs_success": success_jobs,
        "jobs_failed": failed_jobs,
        "raw_rows": int(len(raw_df)),
        "stitched_rows": int(len(stitched_df)),
        "normalized_rows": int(len(normalized_df)),
        "exact_merge_coverage": exact_coverage,
        "observed_rate_by_category": {k: float(v) for k, v in observed_rate.items()},
        "promoted_to_default": bool(promote_to_default),
    }


def main() -> None:
    args = parse_args()
    token = args.token or os.environ.get(args.token_env, "")
    if not token:
        raise SystemExit("Apify token is required. Pass --token or set the environment variable named by --token-env.")

    _, active_categories, min_week, max_week = load_active_categories(args.active_min_weeks)
    if args.categories.strip():
        requested = [item.strip() for item in args.categories.split(",") if item.strip()]
        active_categories = [category for category in active_categories if category in requested]
    if args.max_categories > 0:
        active_categories = active_categories[: args.max_categories]

    categories = [category for category in active_categories if category in CATEGORY_KEYWORDS]
    missing_mapping = sorted(set(active_categories) - set(categories))
    jobs = build_chunk_jobs(categories, min_week, max_week, args.span_weeks, args.overlap_weeks)

    session = build_session()
    existing_raw_df, existing_stitched_df, existing_log_df = load_existing_state()
    raw_frames: list[pd.DataFrame] = [existing_raw_df] if not existing_raw_df.empty else []
    stitched_frames: list[pd.DataFrame] = [existing_stitched_df] if not existing_stitched_df.empty else []
    log_rows: list[dict] = existing_log_df.to_dict(orient="records") if not existing_log_df.empty else []

    jobs_by_category: dict[str, list[ChunkJob]] = {}
    for job in jobs:
        jobs_by_category.setdefault(job.category, []).append(job)

    completed_categories: set[str] = set()
    if not existing_log_df.empty:
        existing_counts = existing_log_df.groupby("category")["chunk_index"].nunique().to_dict()
        for category, category_jobs in jobs_by_category.items():
            if existing_counts.get(category, 0) >= len({job.chunk_index for job in category_jobs}):
                completed_categories.add(category)

    for category, category_jobs in jobs_by_category.items():
        if category in completed_categories:
            print(f"[resume] skip completed category: {category}", flush=True)
            continue

        category_chunk_frames: list[pd.DataFrame] = []
        category_results: list[ChunkResult] = []
        print(f"[collect] category={category} chunks={len(category_jobs)}", flush=True)

        for job in category_jobs:
            frame, result = choose_best_keyword_result(session, token, job, CATEGORY_KEYWORDS[category])
            category_chunk_frames.append(frame)
            category_results.append(result)
            print(
                f"  chunk={job.chunk_index} keyword={result.selected_keyword or '-'} points={result.points} "
                f"status={result.status} error={result.apify_error or result.request_error or '-'}",
                flush=True,
            )
            time.sleep(1.0)

        stitched, scale_log = stitch_category_chunks(category_chunk_frames)
        scale_map = {chunk_index: (scale_factor, overlap_weeks) for chunk_index, scale_factor, overlap_weeks in scale_log}

        for frame in category_chunk_frames:
            if not frame.empty:
                raw_frames.append(frame)

        for result in category_results:
            if result.chunk_index in scale_map:
                result.scale_factor, result.overlap_weeks = scale_map[result.chunk_index]
            log_rows.append(asdict(result))

        if not stitched.empty:
            stitched_frames.append(stitched)

        current_raw = (
            pd.concat(raw_frames, ignore_index=True)
            if raw_frames
            else pd.DataFrame(columns=["category", "keyword", "chunk_index", "query_start", "query_end", "week_date", "search_interest_raw"])
        )
        current_stitched = (
            pd.concat(stitched_frames, ignore_index=True).sort_values(["category", "week_date"]).reset_index(drop=True)
            if stitched_frames
            else pd.DataFrame(columns=["category", "week_date", "search_interest"])
        )
        current_log = pd.DataFrame(log_rows).sort_values(["category", "chunk_index"]).reset_index(drop=True)
        current_normalized = normalize_search_features(current_stitched, categories, min_week, max_week)
        current_summary = build_summary(
            active_min_weeks=args.active_min_weeks,
            categories=categories,
            missing_mapping=missing_mapping,
            min_week=min_week,
            max_week=max_week,
            span_weeks=args.span_weeks,
            overlap_weeks=args.overlap_weeks,
            raw_df=current_raw,
            stitched_df=current_stitched,
            normalized_df=current_normalized,
            log_df=current_log,
            promote_to_default=False,
        )
        save_outputs(current_raw, current_stitched, current_normalized, current_log, current_summary, False)

    raw_df = (
        pd.concat(raw_frames, ignore_index=True)
        if raw_frames
        else pd.DataFrame(columns=["category", "keyword", "chunk_index", "query_start", "query_end", "week_date", "search_interest_raw"])
    )
    stitched_df = (
        pd.concat(stitched_frames, ignore_index=True).sort_values(["category", "week_date"]).reset_index(drop=True)
        if stitched_frames
        else pd.DataFrame(columns=["category", "week_date", "search_interest"])
    )
    normalized_df = normalize_search_features(stitched_df, categories, min_week, max_week)
    log_df = pd.DataFrame(log_rows).sort_values(["category", "chunk_index"]).reset_index(drop=True)
    summary = build_summary(
        active_min_weeks=args.active_min_weeks,
        categories=categories,
        missing_mapping=missing_mapping,
        min_week=min_week,
        max_week=max_week,
        span_weeks=args.span_weeks,
        overlap_weeks=args.overlap_weeks,
        raw_df=raw_df,
        stitched_df=stitched_df,
        normalized_df=normalized_df,
        log_df=log_df,
        promote_to_default=args.promote_to_default,
    )

    save_outputs(raw_df, stitched_df, normalized_df, log_df, summary, args.promote_to_default)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
