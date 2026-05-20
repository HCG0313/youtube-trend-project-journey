from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "project_ready_data" / "model_outputs"
OUT_DIR = ROOT / "project_ready_data" / "ppt_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_PATH = MODEL_DIR / "dl_core10_category_rank_bigru_top_categories.csv"
FUTURE_PATH = MODEL_DIR / "dl_core10_category_rank_bigru_future_probs.csv"

PNG_PATH = OUT_DIR / "core10_prediction_results_refined.png"
PDF_PATH = OUT_DIR / "core10_prediction_results_refined.pdf"


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


def clean_axis(ax, grid_axis: str | None = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#C2CBD0")
    ax.tick_params(axis="y", length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, linestyle="--", linewidth=0.8, alpha=0.75, color="#E7ECEE")


def main() -> None:
    configure_font()

    if not TOP_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TOP_PATH}")
    if not FUTURE_PATH.exists():
        raise FileNotFoundError(f"Missing file: {FUTURE_PATH}")

    top_df = pd.read_csv(TOP_PATH, encoding="utf-8-sig")
    future_df = pd.read_csv(FUTURE_PATH, encoding="utf-8-sig")

    merged = top_df.merge(future_df, on="category", how="left", suffixes=("", "_future"))
    merged = merged.sort_values("final_score", ascending=False).reset_index(drop=True)
    merged["rank"] = np.arange(1, len(merged) + 1)

    # 논문형 figure는 Top-10 전체를 유지하되, 왼쪽은 최종 점수 순위 / 오른쪽은 확률 구성 heatmap으로 정리
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

    BG = "#FFFFFF"
    TEXT = "#1E2A2E"
    SUBTEXT = "#5F6B70"
    BAR = "#355E63"
    BAR_LIGHT = "#DCE6E8"
    TEAL = "#2F5D62"

    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.edgecolor": "#C2CBD0",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.labelcolor": TEXT,
        }
    )

    fig = plt.figure(figsize=(14.2, 7.1), dpi=280)
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.05, 1.15],
        left=0.06,
        right=0.98,
        top=0.84,
        bottom=0.14,
        wspace=0.20,
    )

    # Panel A: final score ranking
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(y, left_df["final_score"], color=BAR, height=0.55, alpha=0.94)
    ax1.barh(y, 1 - left_df["final_score"], left=left_df["final_score"], color=BAR_LIGHT, height=0.55, alpha=0.85)

    for yi, row in enumerate(left_df.itertuples(index=False)):
        ax1.text(
            row.final_score + 0.015,
            yi,
            f"{row.final_score:.3f}",
            va="center",
            ha="left",
            fontsize=9.4,
            color=BAR,
            fontweight="bold",
        )

    ax1.set_yticks(y)
    ax1.set_yticklabels(
        [f"{r}. {c}" for r, c in zip(left_df["rank"], left_df["category"])],
        fontsize=10.8,
        fontweight="bold",
    )
    ax1.set_xlim(0.0, 1.0)
    ax1.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    clean_axis(ax1, grid_axis="x")
    ax1.set_title("A. 최종 추천 순위", loc="left", fontsize=15.5, fontweight="bold", color=TEXT, pad=10)
    ax1.text(
        0.0,
        1.02,
        "상승 확률과 순위 상승 확률을 조합한 최종 점수 기준",
        transform=ax1.transAxes,
        fontsize=10.0,
        color=SUBTEXT,
    )

    # Panel B: probability composition heatmap
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
            ax2.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=9.2,
                color=color,
                fontweight="bold",
            )

    for side in ["top", "right", "left", "bottom"]:
        ax2.spines[side].set_visible(False)
    ax2.tick_params(axis="both", length=0)
    ax2.set_title("B. 추천 분야별 확률 구성", loc="left", fontsize=15.5, fontweight="bold", color=TEXT, pad=10)
    ax2.text(
        0.0,
        1.02,
        "상승 확률, 순위 상승 확률, 그리고 다음 4주 주차별 상승 확률",
        transform=ax2.transAxes,
        fontsize=10.0,
        color=SUBTEXT,
    )

    cbar = fig.colorbar(im, ax=ax2, fraction=0.032, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9.2, colors=TEXT)
    cbar.set_label("예측 확률", fontsize=9.8, color=TEXT)

    # Global title
    fig.text(
        0.06,
        0.95,
        "최종 코어 10개 BiGRU 모델 예측 결과",
        fontsize=20.5,
        fontweight="bold",
        color=TEXT,
        ha="left",
    )
    fig.text(
        0.06,
        0.91,
        "핵심 10개 분야 중 다음 4주 동안 상승 가능성이 높은 분야를 최종 점수와 확률 구성으로 제시",
        fontsize=10.8,
        color=SUBTEXT,
        ha="left",
    )
    fig.text(
        0.06,
        0.05,
        "해석: 반려동물, 먹방, 경제, 브이로그, 교육이 상위권으로 예측되었으며, 최종 추천은 상승 확률과 순위 상승 확률의 결합 신호에 의해 결정되었다.",
        fontsize=10.0,
        color=SUBTEXT,
        ha="left",
    )

    plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.savefig(PDF_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {PNG_PATH}")
    print(f"saved: {PDF_PATH}")


if __name__ == "__main__":
    main()
