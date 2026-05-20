from __future__ import annotations

import json
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
CALENDAR_PATH = EXT_DIR / "korean_weekly_calendar_features.csv"
SUMMARY_PATH = MODEL_DIR / "dl_core10_category_rank_bigru_summary.json"
TOP_PATH = MODEL_DIR / "dl_core10_category_rank_bigru_top_categories.csv"
FUTURE_PATH = MODEL_DIR / "dl_core10_category_rank_bigru_future_probs.csv"


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


BG = "#FFFFFF"
TEXT = "#1E2A2E"
SUBTEXT = "#5F6B70"
GRID = "#E7ECEE"
AXIS = "#BCC7CB"
TEAL = "#2F5D62"
TEAL_LIGHT = "#7DA39D"
SAND = "#C2A88D"
LINK = "#C9D4D7"
MINT = "#B9CCC6"
NAVY = "#355E63"
PALE = "#DCE6E8"


def clean_axis(ax, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, linestyle="--", linewidth=0.8, color=GRID, alpha=0.85)


def save(fig: plt.Figure, stem: str) -> None:
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=BG)
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"saved: {png}")
    print(f"saved: {pdf}")


def bootstrap_mean_diff(holiday_vals, normal_vals, n_boot=1500, seed=42):
    rng = np.random.default_rng(seed)
    holiday_vals = np.asarray(holiday_vals, dtype=float)
    normal_vals = np.asarray(normal_vals, dtype=float)
    boot = []
    for _ in range(n_boot):
        h = rng.choice(holiday_vals, size=len(holiday_vals), replace=True)
        n = rng.choice(normal_vals, size=len(normal_vals), replace=True)
        boot.append(h.mean() - n.mean())
    boot = np.asarray(boot)
    return float(np.mean(boot)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def load_data():
    weekly = pd.read_csv(WEEKLY_PATH, encoding="utf-8-sig")
    search = pd.read_csv(SEARCH_PATH, encoding="utf-8-sig")
    calendar = pd.read_csv(CALENDAR_PATH, encoding="utf-8-sig")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    top_df = pd.read_csv(TOP_PATH, encoding="utf-8-sig")
    future_df = pd.read_csv(FUTURE_PATH, encoding="utf-8-sig")

    weekly["week_date"] = pd.to_datetime(weekly["week_date"])
    search["week_date"] = pd.to_datetime(search["week_date"])
    calendar["week_date"] = pd.to_datetime(calendar["week_date"])

    core10 = [str(x) for x in summary["active_categories"]]
    weekly = weekly[weekly["category"].isin(core10)].copy()
    search = search[search["category"].isin(core10)].copy()

    merged = (
        weekly.merge(search, on=["category", "week_date"], how="left")
        .merge(calendar, on="week_date", how="left")
        .sort_values(["category", "week_date"])
        .reset_index(drop=True)
    )
    merged["search_interest"] = pd.to_numeric(merged["search_interest"], errors="coerce").fillna(0.0)
    merged["avg_virality"] = pd.to_numeric(merged["avg_virality"], errors="coerce").fillna(0.0)
    merged["video_count"] = pd.to_numeric(merged["video_count"], errors="coerce").fillna(0.0)
    merged["log_avg_virality"] = np.log1p(merged["avg_virality"])
    merged["search_roll4"] = (
        merged.groupby("category")["search_interest"]
        .transform(lambda s: s.rolling(4, min_periods=1).mean())
    )
    merged["response_index"] = np.nan

    for category, idx in merged.groupby("category").groups.items():
        sub = merged.loc[idx].copy()
        baseline = float(sub.head(min(4, len(sub)))["avg_virality"].replace(0, np.nan).mean())
        if not np.isfinite(baseline) or baseline <= 0:
            baseline = float(sub["avg_virality"].replace(0, np.nan).median())
        if not np.isfinite(baseline) or baseline <= 0:
            baseline = 1.0
        merged.loc[idx, "response_index"] = 100.0 * sub["avg_virality"] / baseline

    return merged, summary, top_df, future_df, core10


def make_eda_recent_trends(merged: pd.DataFrame, core10: list[str]) -> None:
    recent_cut = merged["week_date"].max() - pd.Timedelta(weeks=104)
    plot_df = merged.loc[merged["week_date"] >= recent_cut].copy()
    palette = {
        cat: col
        for cat, col in zip(
            core10,
            ["#2F5D62", "#577E7A", "#6F8D89", "#84A29D", "#9BB7B1", "#B39C7D", "#C2A88D", "#D2B48C", "#89A1A5", "#AEBFC3"],
        )
    }

    fig, axes = plt.subplots(2, 5, figsize=(18, 8.8), dpi=240, sharex=True)
    axes = axes.flatten()
    for ax, category in zip(axes, core10):
        sub = plot_df.loc[plot_df["category"] == category].sort_values("week_date")
        ax.plot(sub["week_date"], sub["avg_virality"], color="#D9E0E3", linewidth=1.2, alpha=0.9)
        ax.plot(sub["week_date"], sub["rolling_4week_mean"], color=palette[category], linewidth=2.4)
        ax.set_title(category, fontsize=11.5, color=TEXT, pad=8, fontweight="bold")
        clean_axis(ax, grid_axis="y")
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("핵심 10개 분야 최근 추세", x=0.065, y=0.98, ha="left", fontsize=21, fontweight="bold", color=TEXT)
    fig.text(0.065, 0.94, "최근 2년 구간의 평균 반응 규모와 4주 이동평균을 비교하여 분야별 흐름 차이를 확인", fontsize=11, color=SUBTEXT)
    plt.tight_layout(rect=[0.03, 0.04, 0.98, 0.91])
    save(fig, "paper_eda_core10_recent_trends")


def make_eda_intensity_heatmap(merged: pd.DataFrame, core10: list[str]) -> None:
    recent_1y = merged.loc[merged["week_date"] >= merged["week_date"].max() - pd.Timedelta(weeks=52)].copy()
    recent_1y["within_cat_z"] = (
        recent_1y.groupby("category")["log_avg_virality"]
        .transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    )
    heat = (
        recent_1y.pivot_table(index="category", columns="week_date", values="within_cat_z", aggfunc="mean")
        .reindex(core10)
    )

    fig, ax = plt.subplots(figsize=(15.5, 6.8), dpi=240)
    im = ax.imshow(heat.values, aspect="auto", cmap="RdBu_r", vmin=-2.2, vmax=2.2, interpolation="nearest")
    date_cols = pd.to_datetime(heat.columns)
    tick_idx = np.linspace(0, len(date_cols) - 1, 8, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([date_cols[i].strftime("%Y-%m") for i in tick_idx], fontsize=9)
    ax.set_yticks(np.arange(len(core10)))
    ax.set_yticklabels(core10, fontsize=10.5, color=TEXT, fontweight="bold")
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_title("핵심 10개 분야 최근 1년 반응 강도 Heatmap", loc="left", fontsize=18, fontweight="bold", color=TEXT, pad=12)
    ax.text(0.0, 1.03, "각 셀은 카테고리 내부 기준으로 표준화된 반응 강도(z-score)를 의미", transform=ax.transAxes, fontsize=10.5, color=SUBTEXT)
    cbar = fig.colorbar(im, ax=ax, pad=0.015, shrink=0.95)
    cbar.outline.set_visible(False)
    cbar.set_label("카테고리 내부 표준화 반응 강도", color=TEXT, fontsize=10)
    plt.tight_layout()
    save(fig, "paper_eda_core10_intensity_heatmap")


def make_eda_holiday_effect(merged: pd.DataFrame) -> None:
    recent_2y = merged.loc[merged["week_date"] >= merged["week_date"].max() - pd.Timedelta(weeks=104)].copy()
    recent_2y["log_avg_virality"] = np.log1p(recent_2y["avg_virality"])
    rows = []
    for category, sub in recent_2y.groupby("category"):
        holiday_vals = sub.loc[sub["is_holiday_week"] > 0, "log_avg_virality"].dropna()
        normal_vals = sub.loc[sub["is_holiday_week"] == 0, "log_avg_virality"].dropna()
        if len(holiday_vals) >= 3 and len(normal_vals) >= 6:
            mean_diff, low, high = bootstrap_mean_diff(holiday_vals, normal_vals)
            rows.append([category, mean_diff, low, high, len(holiday_vals)])
    coef_df = pd.DataFrame(rows, columns=["category", "effect", "ci_low", "ci_high", "n_holiday"]).sort_values("effect")
    y = np.arange(len(coef_df))
    fig, ax = plt.subplots(figsize=(11.8, 6.8), dpi=240)
    ax.hlines(y, coef_df["ci_low"], coef_df["ci_high"], color=AXIS, linewidth=2.2, zorder=1)
    ax.scatter(coef_df["effect"], y, s=95, color=TEAL, edgecolor="white", linewidth=1.1, zorder=3)
    ax.axvline(0, color=SAND, linestyle="--", linewidth=1.4)
    ax.set_yticks(y)
    ax.set_yticklabels(coef_df["category"], fontsize=10.5, color=TEXT, fontweight="bold")
    ax.set_xlabel("공휴일 주간 효과 추정치 (log 반응 차이)", fontsize=11.5, color=TEXT)
    clean_axis(ax, grid_axis="x")
    ax.set_title("핵심 10개 분야의 공휴일 효과 추정치", loc="left", fontsize=18, fontweight="bold", color=TEXT, pad=12)
    ax.text(0.0, 1.03, "최근 2년 구간에서 공휴일 주간과 비공휴일 주간의 평균 반응 차이를 추정", transform=ax.transAxes, fontsize=10.5, color=SUBTEXT)
    plt.tight_layout()
    save(fig, "paper_eda_core10_holiday_effect")


def make_performance_figure(summary: dict) -> None:
    val_metrics = summary["val_metrics"]
    test_metrics = summary["test_metrics"]
    rank_metrics = summary["test_ranking_metrics"]
    cls_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Balanced Accuracy", "F1-score", "ROC AUC"],
            "Validation": [
                val_metrics["accuracy"],
                val_metrics["balanced_accuracy"],
                val_metrics["f1"],
                val_metrics["roc_auc"],
            ],
            "Test": [
                test_metrics["accuracy"],
                test_metrics["balanced_accuracy"],
                test_metrics["f1"],
                test_metrics["roc_auc"],
            ],
        }
    )
    rank_df = pd.DataFrame(
        {
            "Metric": ["Precision@5", "Recall@5", "HitRate@5", "NDCG@5"],
            "Score": [
                rank_metrics["precision_at_5"],
                rank_metrics["recall_at_5"],
                rank_metrics["hit_rate_at_5"],
                rank_metrics["ndcg_at_5"],
            ],
        }
    )

    fig = plt.figure(figsize=(13.0, 6.3), dpi=260)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9], left=0.06, right=0.98, top=0.82, bottom=0.14, wspace=0.24)

    ax1 = fig.add_subplot(gs[0, 0])
    y = np.arange(len(cls_df))
    for i, row in cls_df.iterrows():
        ax1.plot([row["Validation"], row["Test"]], [i, i], color=LINK, linewidth=2.0, zorder=1)
    ax1.scatter(cls_df["Validation"], y, s=85, color=TEAL_LIGHT, edgecolor="white", linewidth=1.0, zorder=3)
    ax1.scatter(cls_df["Test"], y, s=90, color=TEAL, edgecolor="white", linewidth=1.0, zorder=3)
    for i, row in cls_df.iterrows():
        ax1.text(row["Validation"] - 0.007, i + 0.16, f"{row['Validation']:.3f}", fontsize=9.0, color=TEAL_LIGHT, ha="right", va="center", fontweight="bold")
        ax1.text(row["Test"] + 0.007, i + 0.16, f"{row['Test']:.3f}", fontsize=9.0, color=TEAL, ha="left", va="center", fontweight="bold")
    ax1.set_yticks(y)
    ax1.set_yticklabels(cls_df["Metric"], fontsize=11, fontweight="bold")
    ax1.set_xlim(0.66, 0.88)
    ax1.set_xticks([0.70, 0.75, 0.80, 0.85])
    ax1.invert_yaxis()
    clean_axis(ax1, grid_axis="x")
    ax1.set_title("A. 분류 성능", loc="left", fontsize=15, fontweight="bold", color=TEXT, pad=10)
    ax1.text(0.0, 1.02, "Validation-Test 비교", transform=ax1.transAxes, fontsize=10.0, color=SUBTEXT)

    ax2 = fig.add_subplot(gs[0, 1])
    y2 = np.arange(len(rank_df))
    for yi, score in zip(y2, rank_df["Score"]):
        ax2.hlines(yi, 0.70, 1.00, color=PALE, linewidth=7, zorder=1, capstyle="round")
        ax2.hlines(yi, 0.70, score, color=TEAL, linewidth=7, zorder=2, capstyle="round")
        ax2.scatter(score, yi, s=145, color=TEAL, edgecolor="white", linewidth=1.1, zorder=3)
        ax2.text(score + 0.008, yi, f"{score:.3f}", fontsize=9.6, color=TEAL, va="center", ha="left", fontweight="bold")
    ax2.set_yticks(y2)
    ax2.set_yticklabels(rank_df["Metric"], fontsize=11, fontweight="bold")
    ax2.set_xlim(0.70, 1.02)
    ax2.set_xticks([0.70, 0.80, 0.90, 1.00])
    ax2.invert_yaxis()
    clean_axis(ax2, grid_axis="x")
    ax2.set_title("B. Top-5 선별 성능", loc="left", fontsize=15, fontweight="bold", color=TEXT, pad=10)
    ax2.text(0.0, 1.02, "Test 기준", transform=ax2.transAxes, fontsize=10.0, color=SUBTEXT)

    fig.text(0.06, 0.95, "최종 코어 10개 BiGRU 모델 성능 평가", fontsize=20, fontweight="bold", color=TEXT, ha="left")
    fig.text(0.06, 0.91, "분류 성능과 Top-5 선별 성능을 함께 제시하여 상승 분야 예측 성능을 평가", fontsize=10.8, color=SUBTEXT, ha="left")
    fig.text(0.06, 0.05, "해석: 코어 10개 분야 기준으로 분류 성능이 안정적으로 나타났으며, Precision@5와 NDCG@5가 높아 Top-5 상승 분야 선별에 효과적임을 확인하였다.", fontsize=10.0, color=SUBTEXT, ha="left")
    save(fig, "paper_results_core10_performance")


def make_prediction_rank_heatmap(top_df: pd.DataFrame, future_df: pd.DataFrame) -> None:
    merged = top_df.merge(future_df, on="category", how="left", suffixes=("", "_future"))
    merged = merged.sort_values("final_score", ascending=False).reset_index(drop=True)
    merged["rank"] = np.arange(1, len(merged) + 1)
    left_df = merged.copy().sort_values("final_score", ascending=True)
    y = np.arange(len(left_df))

    heat_cols = [
        "rise_probability",
        "rank_up_probability",
        "step1_up_probability",
        "step2_up_probability",
        "step3_up_probability",
        "step4_up_probability",
    ]
    heat_labels = ["Rise", "Rank-up", "t+1", "t+2", "t+3", "t+4"]
    heat_df = merged.set_index("category")[heat_cols]

    fig = plt.figure(figsize=(14.2, 7.1), dpi=280)
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.05, 1.15],
        left=0.06,
        right=0.98,
        top=0.82,
        bottom=0.14,
        wspace=0.20,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(y, left_df["final_score"], color=NAVY, height=0.55, alpha=0.94)
    ax1.barh(y, 1 - left_df["final_score"], left=left_df["final_score"], color=PALE, height=0.55, alpha=0.85)
    for yi, row in enumerate(left_df.itertuples(index=False)):
        ax1.text(row.final_score + 0.015, yi, f"{row.final_score:.3f}", va="center", ha="left", fontsize=9.4, color=NAVY, fontweight="bold")
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{r}. {c}" for r, c in zip(left_df["rank"], left_df["category"])], fontsize=10.8, fontweight="bold")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    clean_axis(ax1, grid_axis="x")
    ax1.set_title(
        "A. 최종 상승 예측 순위",
        loc="left",
        fontsize=14.2,
        fontweight="bold",
        color=TEXT,
        pad=6,
    )

    ax2 = fig.add_subplot(gs[0, 1])
    mat = heat_df.to_numpy(dtype=float)
    im = ax2.imshow(mat, cmap="PuBuGn", aspect="auto", vmin=0.20, vmax=0.70)
    ax2.set_xticks(np.arange(len(heat_labels)))
    ax2.set_xticklabels(heat_labels, fontsize=10.5, fontweight="bold")
    ax2.set_yticks(np.arange(len(heat_df.index)))
    ax2.set_yticklabels(heat_df.index.tolist(), fontsize=10.8, fontweight="bold")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            color = "white" if value >= 0.48 else TEXT
            ax2.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=9.2, color=color, fontweight="bold")
    for side in ["top", "right", "left", "bottom"]:
        ax2.spines[side].set_visible(False)
    ax2.tick_params(axis="both", length=0)
    ax2.set_title(
        "B. 분야별 확률 구성",
        loc="left",
        fontsize=14.2,
        fontweight="bold",
        color=TEXT,
        pad=6,
    )
    cbar = fig.colorbar(im, ax=ax2, fraction=0.032, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9.2, colors=TEXT)
    cbar.set_label("예측 확률", fontsize=9.8, color=TEXT)

    fig.text(0.06, 0.95, "최종 코어 10개 BiGRU 모델 예측 결과", fontsize=19.2, fontweight="bold", color=TEXT, ha="left")
    fig.text(0.06, 0.905, "핵심 10개 분야 중 다음 4주 동안 상승 가능성이 높은 분야를 최종 점수와 확률 구성으로 제시", fontsize=10.4, color=SUBTEXT, ha="left")
    fig.text(0.06, 0.05, "해석: 반려동물, 먹방, 경제, 브이로그, 교육이 상위권으로 예측되었으며, 최종 순위는 상승 확률과 순위 상승 확률의 결합 신호에 의해 결정되었다.", fontsize=9.8, color=SUBTEXT, ha="left")
    save(fig, "paper_results_core10_prediction_rank_heatmap")


def make_prediction_story(merged: pd.DataFrame, top_df: pd.DataFrame) -> None:
    top5 = top_df.sort_values("final_score", ascending=False).head(5).copy()
    cat_order = top5["category"].tolist()
    score_map = dict(zip(top5["category"], top5["final_score"]))
    story = merged[merged["category"].isin(cat_order)].copy()
    story = story.loc[story["week_date"] >= story["week_date"].max() - pd.Timedelta(weeks=26)].copy()
    story["response_roll4_index"] = (
        story.groupby("category")["response_index"]
        .transform(lambda s: s.rolling(4, min_periods=1).mean())
    )
    clip_upper = float(np.nanpercentile(story["response_roll4_index"], 95))
    story["response_roll4_index_clipped"] = story["response_roll4_index"].clip(upper=clip_upper)
    palette = {
        cat_order[0]: "#2F5D62",
        cat_order[1]: "#577E7A",
        cat_order[2]: "#8FA8A2",
        cat_order[3]: "#B39C7D",
        cat_order[4]: "#D2B48C",
    }

    fig, axes = plt.subplots(3, 1, figsize=(10.8, 8.1), dpi=280, sharex=True)
    plt.subplots_adjust(left=0.09, right=0.92, top=0.80, bottom=0.12, hspace=0.34)

    ax = axes[0]
    for category in cat_order:
        sub = story.loc[story["category"] == category].sort_values("week_date")
        ax.plot(sub["week_date"], sub["response_roll4_index_clipped"], color=palette[category], linewidth=2.0, alpha=0.95)
        ax.scatter(sub["week_date"].iloc[-1], sub["response_roll4_index_clipped"].iloc[-1], s=22, color=palette[category], zorder=3)
        ax.text(
            sub["week_date"].iloc[-1] + pd.Timedelta(days=4),
            sub["response_roll4_index_clipped"].iloc[-1],
            f"{category}  {score_map[category]:.3f}",
            fontsize=8.3,
            color=palette[category],
            va="center",
        )
    clean_axis(ax)
    ax.set_title("A. 최근 반응 추세", loc="left", fontsize=11.8, fontweight="bold", color=TEXT, pad=4)
    ax.set_ylabel("반응 지수", fontsize=9.2, color=TEXT)

    ax = axes[1]
    for category in cat_order:
        sub = story.loc[story["category"] == category].sort_values("week_date")
        ax.plot(sub["week_date"], sub["search_roll4"], color=palette[category], linewidth=2.0, alpha=0.95)
        ax.scatter(sub["week_date"].iloc[-1], sub["search_roll4"].iloc[-1], s=22, color=palette[category], zorder=3)
    clean_axis(ax)
    ax.set_title("B. 검색 관심도 추세", loc="left", fontsize=11.8, fontweight="bold", color=TEXT, pad=4)
    ax.set_ylabel("검색 관심도", fontsize=9.2, color=TEXT)

    ax = axes[2]
    for category in cat_order:
        sub = story.loc[story["category"] == category].sort_values("week_date")
        ax.plot(sub["week_date"], sub["category_rank"], color=palette[category], linewidth=2.0, alpha=0.95)
        ax.scatter(sub["week_date"].iloc[-1], sub["category_rank"].iloc[-1], s=22, color=palette[category], zorder=3)
    clean_axis(ax)
    ax.set_title("C. 카테고리 순위 추세", loc="left", fontsize=11.8, fontweight="bold", color=TEXT, pad=4)
    ax.set_ylabel("카테고리 순위", fontsize=9.2, color=TEXT)
    ax.set_xlabel("주차", fontsize=9.4, color=TEXT)
    ax.invert_yaxis()

    fig.text(0.09, 0.95, "최종 Top-5 상승 예측 분야의 최근 추세", fontsize=18.0, fontweight="bold", color=TEXT, ha="left")
    fig.text(0.09, 0.905, "반려동물, 먹방, 경제, 브이로그, 교육이 최종 상위권으로 예측되었으며, 최근 반응과 외부 관심도 흐름을 함께 제시", fontsize=10.0, color=SUBTEXT, ha="left")
    fig.text(0.09, 0.04, "해석: 최종 상위 분야는 단순 절대 규모가 아니라 최근 반응의 유지·회복 패턴, 검색 관심도, 상대 순위 신호를 함께 반영한 결과이다.", fontsize=9.6, color=SUBTEXT, ha="left")
    save(fig, "paper_results_core10_prediction_story")


def make_category_profile(top_df: pd.DataFrame) -> None:
    profile = top_df.sort_values("final_score", ascending=True).copy()
    y = np.arange(len(profile))

    fig, ax = plt.subplots(figsize=(11.6, 7.0), dpi=280)
    for i, row in enumerate(profile.itertuples(index=False)):
        low = min(row.rise_probability, row.rank_up_probability)
        high = max(row.rise_probability, row.rank_up_probability)
        ax.plot([low, high], [i, i], color=LINK, linewidth=2.0, zorder=1)
        ax.scatter(row.rise_probability, i, s=85, color=TEAL_LIGHT, edgecolor="white", linewidth=1.0, zorder=3)
        ax.scatter(row.rank_up_probability, i, s=85, color=SAND, edgecolor="white", linewidth=1.0, zorder=3)
        ax.scatter(row.final_score, i, s=95, marker="D", color=TEAL, edgecolor="white", linewidth=1.0, zorder=4)
        ax.text(row.final_score + 0.008, i + 0.14, f"{row.final_score:.3f}", fontsize=9.1, color=TEAL, ha="left", va="center", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(profile["category"], fontsize=11, fontweight="bold")
    ax.set_xlim(0.20, 0.60)
    ax.set_xticks([0.20, 0.30, 0.40, 0.50, 0.60])
    clean_axis(ax, grid_axis="x")
    ax.text(0.0, 1.01, "원형은 상승 확률, 점은 순위 상승 확률, 마름모는 최종 점수를 의미", transform=ax.transAxes, fontsize=9.6, color=SUBTEXT)
    ax.text(0.98, 0.98, "상승 확률", transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=TEAL_LIGHT)
    ax.text(0.98, 0.93, "순위 상승 확률", transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=SAND)
    ax.text(0.98, 0.88, "최종 점수", transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=TEAL)
    fig.text(0.125, 0.955, "핵심 10개 분야별 예측 확률 프로파일", fontsize=17.5, fontweight="bold", color=TEXT, ha="left")
    fig.text(0.125, 0.905, "각 핵심 분야의 상승 확률, 순위 상승 확률, 최종 점수를 동시에 비교", fontsize=9.8, color=SUBTEXT, ha="left")
    fig.text(0.125, 0.03, "해석: 반려동물, 먹방, 경제, 브이로그, 교육은 상승 확률과 순위 상승 확률의 조합에서 상대적으로 높은 값을 보이며 최종 상위권으로 분류되었다.", fontsize=9.8, color=SUBTEXT)
    plt.tight_layout(rect=[0.03, 0.06, 0.98, 0.88])
    save(fig, "paper_results_core10_category_profile")


def make_step_probability_heatmap(future_df: pd.DataFrame) -> None:
    step_cols = ["step1_up_probability", "step2_up_probability", "step3_up_probability", "step4_up_probability"]
    step_labels = ["t+1", "t+2", "t+3", "t+4"]
    heat_df = future_df.sort_values("ranking_score", ascending=False).set_index("category")[step_cols]
    fig, ax = plt.subplots(figsize=(9.2, 6.2), dpi=280)
    im = ax.imshow(heat_df.values, cmap="YlGnBu", aspect="auto", vmin=0.25, vmax=0.70)
    ax.set_xticks(np.arange(len(step_labels)))
    ax.set_xticklabels(step_labels, fontsize=10.5, fontweight="bold")
    ax.set_yticks(np.arange(len(heat_df.index)))
    ax.set_yticklabels(heat_df.index.tolist(), fontsize=10.8, fontweight="bold")
    for i in range(heat_df.shape[0]):
        for j in range(heat_df.shape[1]):
            value = heat_df.iloc[i, j]
            color = "white" if value >= 0.48 else TEXT
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=9.2, color=color, fontweight="bold")
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", length=0)
    fig.text(0.125, 0.955, "핵심 10개 분야의 4주 선행 상승 확률", fontsize=18.0, fontweight="bold", color=TEXT, ha="left")
    fig.text(0.125, 0.91, "각 카테고리의 다음 4주 주차별 상승 확률을 heatmap으로 비교", fontsize=10.0, color=SUBTEXT, ha="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.025)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9.2, colors=TEXT)
    cbar.set_label("상승 확률", fontsize=9.8, color=TEXT)
    plt.tight_layout(rect=[0.04, 0.04, 0.98, 0.88])
    save(fig, "paper_results_core10_step_probability_heatmap")


def main() -> None:
    configure_font()
    merged, summary, top_df, future_df, core10 = load_data()
    make_eda_recent_trends(merged, core10)
    make_eda_intensity_heatmap(merged, core10)
    make_eda_holiday_effect(merged)
    make_performance_figure(summary)
    make_prediction_rank_heatmap(top_df, future_df)
    make_prediction_story(merged, top_df)
    make_category_profile(top_df)
    make_step_probability_heatmap(future_df)


if __name__ == "__main__":
    main()
