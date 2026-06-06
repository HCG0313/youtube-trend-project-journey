import math
from pathlib import Path

import pandas as pd

from week_utils import iso_year_week_from_series


ROOT = Path(__file__).resolve().parent


def load_historical_frames(directory: Path):
    if not directory.exists():
        return []
    return [pd.read_csv(path) for path in sorted(directory.glob("historical_videos_offset_*.csv"))]


def compute_virality(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["final_view_count", "final_like_count", "final_comment_count"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)
    work["published_at"] = pd.to_datetime(work["published_at"], utc=True, errors="coerce")
    work["virality_score"] = (work["final_view_count"] + 2 * work["final_like_count"] + 1).map(math.log)
    work["year_week"] = iso_year_week_from_series(work["published_at"])
    return work


def build_trend_dataset(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["category", "year_week"], dropna=False)
        .agg(
            video_count=("video_id", "count"),
            avg_virality=("virality_score", "mean"),
            median_virality=("virality_score", "median"),
            avg_views=("final_view_count", "mean"),
            avg_likes=("final_like_count", "mean"),
        )
        .reset_index()
        .sort_values(["category", "year_week"])
    )

    grouped["lag1_avg_virality"] = grouped.groupby("category")["avg_virality"].shift(1)
    grouped["lag2_avg_virality"] = grouped.groupby("category")["avg_virality"].shift(2)
    grouped["lag1_video_count"] = grouped.groupby("category")["video_count"].shift(1)
    grouped["target_next_avg_virality"] = grouped.groupby("category")["avg_virality"].shift(-1)
    return grouped


def main():
    historical_dir = ROOT / "youtube_category_output_historical"
    output_dir = ROOT / "youtube_category_output_historical"
    frames = load_historical_frames(historical_dir)
    if not frames:
        raise SystemExit("No historical video CSV files found.")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["video_id"], keep="last")
    merged = compute_virality(merged)
    trend_df = build_trend_dataset(merged)

    merged_path = output_dir / "historical_videos_merged.csv"
    trend_path = output_dir / "category_trend_dataset.csv"
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    trend_df.to_csv(trend_path, index=False, encoding="utf-8-sig")

    print(f"Saved merged historical videos to {merged_path}")
    print(f"Saved category trend dataset to {trend_path}")
    print(
        {
            "historical_rows": len(merged),
            "trend_rows": len(trend_df),
            "categories": int(trend_df["category"].nunique()),
        }
    )


if __name__ == "__main__":
    main()
