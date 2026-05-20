from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "project_ready_data"
MODEL_DIR = DATA_DIR / "model_outputs"
EXT_DIR = DATA_DIR / "external_features"
OUT_DIR = DATA_DIR / "ppt_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEEKLY_PATH = DATA_DIR / "ready_category_weekly_trend.csv"
SEARCH_PATH = EXT_DIR / "google_trends_weekly_active_categories_normalized.csv"
TOP_PATH = MODEL_DIR / "dl_core10_category_rank_bigru_top_categories.csv"

PNG_PATH = OUT_DIR / "core10_prediction_story.png"
PDF_PATH = OUT_DIR / "core10_prediction_story.pdf"


def configure_font() -> None:
    for font_path in [
        Path(r"C:\Windows\Fonts\malgun.ttf"),
        Path(r"C:\Windows\Fonts\malgunbd.ttf"),
    ]:
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = fm.FontProperties(fname=str(font_path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#C3CCD0")
    ax.tick_params(axis="y", length=0, labelsize=8.8)
    ax.tick_params(axis="x", labelsize=8.6)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.78, color="#E8ECEE")


def main() -> None:
    configure_font()

    weekly = pd.read_csv(WEEKLY_PATH, encoding="utf-8-sig")
    search = pd.read_csv(SEARCH_PATH, encoding="utf-8-sig")
    top_df = pd.read_csv(TOP_PATH, encoding="utf-8-sig")

    weekly["week_date"] = pd.to_datetime(weekly["week_date"])
    search["week_date"] = pd.to_datetime(search["week_date"])

    top5 = top_df.sort_values("final_score", ascending=False).head(5).copy()
    top5_categories = top5["category"].tolist()
    score_map = dict(zip(top5["category"], top5["final_score"]))

    weekly = weekly[weekly["category"].isin(top5_categories)].copy()
    search = search[search["category"].isin(top5_categories)].copy()

    merged = weekly.merge(
        search[["category", "week_date", "search_interest"]],
        on=["category", "week_date"],
        how="left",
    )
    merged = merged.sort_values(["category", "week_date"]).reset_index(drop=True)

    latest_week = merged["week_date"].max()
    start_cut = latest_week - pd.Timedelta(weeks=26)
    merged = merged.loc[merged["week_date"] >= start_cut].copy()

    # 상대 지수화된 반응 규모
    merged["response_index"] = np.nan
    for category, idx in merged.groupby("category").groups.items():
        sub = merged.loc[idx].copy()
        baseline = float(sub.head(min(4, len(sub)))["avg_virality"].replace(0, np.nan).mean())
        if not np.isfinite(baseline) or baseline <= 0:
            baseline = float(sub["avg_virality"].replace(0, np.nan).median())
        if not np.isfinite(baseline) or baseline <= 0:
            baseline = 1.0
        merged.loc[idx, "response_index"] = 100.0 * sub["avg_virality"] / baseline

    merged["search_roll4"] = (
        merged.groupby("category")["search_interest"]
        .transform(lambda s: s.rolling(4, min_periods=1).mean())
    )

    # rank는 낮을수록 상위이므로, 보기 좋게 invert
    merged["rank_plot"] = merged["category_rank"]

    # plotting order = final score descending
    cat_order = top5_categories
    palette = {
        cat_order[0]: "#2F5D62",
        cat_order[1]: "#577E7A",
        cat_order[2]: "#8FA8A2",
        cat_order[3]: "#B39C7D",
        cat_order[4]: "#D2B48C",
    }

    fig, axes = plt.subplots(3, 1, figsize=(10.8, 8.1), dpi=280, sharex=True)
    plt.subplots_adjust(left=0.09, right=0.92, top=0.85, bottom=0.12, hspace=0.32)

    # Panel 1: response index
    ax = axes[0]
    for category in cat_order:
        sub = merged.loc[merged["category"] == category].sort_values("week_date")
        ax.plot(sub["week_date"], sub["response_index"], color=palette[category], linewidth=2.0, alpha=0.95)
        ax.scatter(sub["week_date"].iloc[-1], sub["response_index"].iloc[-1], s=22, color=palette[category], zorder=3)
        ax.text(
            sub["week_date"].iloc[-1] + pd.Timedelta(days=4),
            sub["response_index"].iloc[-1],
            f"{category}  {score_map[category]:.3f}",
            fontsize=8.3,
            color=palette[category],
            va="center",
        )
    clean_axis(ax)
    ax.set_title("최종 Top-5 상승 예측 분야의 최근 반응 추세", loc="left", fontsize=12.8, fontweight="bold", color="#1E2A2E", pad=8)
    ax.text(0.0, 1.03, "초기 4주 평균을 100으로 둔 상대 반응 지수", transform=ax.transAxes, fontsize=9.1, color="#5F6B70")
    ax.set_ylabel("반응 지수", fontsize=9.2, color="#1E2A2E")

    # Panel 2: search trend
    ax = axes[1]
    for category in cat_order:
        sub = merged.loc[merged["category"] == category].sort_values("week_date")
        ax.plot(sub["week_date"], sub["search_roll4"], color=palette[category], linewidth=2.0, alpha=0.95)
        ax.scatter(sub["week_date"].iloc[-1], sub["search_roll4"].iloc[-1], s=22, color=palette[category], zorder=3)
    clean_axis(ax)
    ax.set_title("최종 Top-5 상승 예측 분야의 검색 관심도 추세", loc="left", fontsize=12.8, fontweight="bold", color="#1E2A2E", pad=8)
    ax.text(0.0, 1.03, "Google Trends 4주 이동평균", transform=ax.transAxes, fontsize=9.1, color="#5F6B70")
    ax.set_ylabel("검색 관심도", fontsize=9.2, color="#1E2A2E")

    # Panel 3: category rank trend
    ax = axes[2]
    for category in cat_order:
        sub = merged.loc[merged["category"] == category].sort_values("week_date")
        ax.plot(sub["week_date"], sub["rank_plot"], color=palette[category], linewidth=2.0, alpha=0.95)
        ax.scatter(sub["week_date"].iloc[-1], sub["rank_plot"].iloc[-1], s=22, color=palette[category], zorder=3)
    clean_axis(ax)
    ax.set_title("최종 Top-5 상승 예측 분야의 카테고리 순위 추세", loc="left", fontsize=12.8, fontweight="bold", color="#1E2A2E", pad=8)
    ax.text(0.0, 1.03, "낮을수록 상위 순위를 의미", transform=ax.transAxes, fontsize=9.1, color="#5F6B70")
    ax.set_ylabel("카테고리 순위", fontsize=9.2, color="#1E2A2E")
    ax.set_xlabel("주차", fontsize=9.4, color="#1E2A2E")
    ax.invert_yaxis()

    fig.text(
        0.09,
        0.94,
        "최종 Top-5 상승 예측 분야의 최근 추세",
        fontsize=19.0,
        fontweight="bold",
        color="#1E2A2E",
        ha="left",
    )
    fig.text(
        0.09,
        0.90,
        "반려동물, 먹방, 경제, 브이로그, 교육이 최종 상위권으로 예측되었으며, 최근 반응과 외부 관심도 흐름을 함께 제시",
        fontsize=10.2,
        color="#5F6B70",
        ha="left",
    )
    fig.text(
        0.09,
        0.04,
        "해석: 최종 상위 분야는 단순 절대 규모가 아니라 최근 반응의 유지·회복 패턴, 검색 관심도, 상대 순위 신호를 함께 반영한 결과이다.",
        fontsize=9.6,
        color="#5F6B70",
        ha="left",
    )

    plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.savefig(PDF_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {PNG_PATH}")
    print(f"saved: {PDF_PATH}")


if __name__ == "__main__":
    main()
