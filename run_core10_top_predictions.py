from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

import train_active_category_rank_bigru as rank_mod


def detect_root() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        # Jupyter/interactive execution path.
        return Path.cwd()


ROOT = detect_root()
MODEL_DIR = ROOT / "project_ready_data" / "model_outputs"
EXT_DIR = ROOT / "project_ready_data" / "external_features"

SUMMARY_PATH = MODEL_DIR / "dl_core10_category_rank_bigru_summary.json"
TOP_PATH = MODEL_DIR / "dl_core10_category_rank_bigru_top_categories.csv"
FUTURE_PATH = MODEL_DIR / "dl_core10_category_rank_bigru_future_probs.csv"
METRIC_PATH = MODEL_DIR / "dl_core10_category_rank_bigru_reproduced_test_metrics.json"

SEARCH_NORMALIZED_PATH = EXT_DIR / "google_trends_weekly_active_categories_normalized.csv"
SEARCH_CACHE_PATH = EXT_DIR / "google_trends_weekly_active_categories.csv"


def load_cached_search_features(
    active_categories: list[str],
    min_week: pd.Timestamp,
    max_week: pd.Timestamp,
) -> pd.DataFrame:
    if SEARCH_NORMALIZED_PATH.exists():
        search_df = pd.read_csv(SEARCH_NORMALIZED_PATH, parse_dates=["week_date"])
        search_df = search_df[search_df["category"].isin(active_categories)].copy()
        return search_df

    if SEARCH_CACHE_PATH.exists():
        raw_search = pd.read_csv(SEARCH_CACHE_PATH, parse_dates=["week_date"])
        return rank_mod.normalize_search_features(raw_search, active_categories, min_week, max_week)

    return rank_mod.normalize_search_features(pd.DataFrame(), active_categories, min_week, max_week)


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Summary file not found: {SUMMARY_PATH}")

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    selected_categories = [str(x) for x in summary["active_categories"]]
    feature_mode = str(summary["feature_mode"])
    quantile = float(summary["quantile"])
    fixed_threshold = float(summary["threshold"])

    blend = summary["blend_weights"]
    blend_weights = (
        float(blend["ranking_score"]),
        float(blend["rise_probability"]),
        float(blend["rank_up_probability"]),
    )

    rank_mod.set_seed(rank_mod.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = pd.read_csv(rank_mod.DATA_PATH, encoding="utf-8-sig")
    recent_activity = rank_mod.build_recent_activity_counts(raw)

    active_categories = [
        category
        for category in selected_categories
        if recent_activity.get(category, 0) >= rank_mod.ACTIVE_MIN_WEEKS
    ]

    if len(active_categories) != len(selected_categories):
        missing = sorted(set(selected_categories) - set(active_categories))
        raise RuntimeError(
            f"Some core10 categories are not eligible under ACTIVE_MIN_WEEKS: {missing}"
        )

    raw["week_date"] = raw["year_week"].map(rank_mod.year_week_to_date)
    min_week = pd.Timestamp(raw["week_date"].min())
    max_week = pd.Timestamp(raw["week_date"].max())

    calendar_df = rank_mod.build_calendar_features(
        sorted(raw["week_date"].dropna().unique().tolist())
    )
    search_df = load_cached_search_features(active_categories, min_week, max_week)

    dense, category_to_idx, has_search = rank_mod.build_dense_active_grid(
        raw,
        active_categories,
        calendar_df,
        search_df,
    )
    arrays = rank_mod.build_rise_windows(dense, rank_mod.SEQ_LEN, rank_mod.HORIZON)

    train_weeks, val_weeks, test_weeks = rank_mod.build_start_week_splits(arrays.start_week)
    train_idx = rank_mod.indices_for_weeks(arrays.start_week, train_weeks)
    val_idx = rank_mod.indices_for_weeks(arrays.start_week, val_weeks)
    test_idx = rank_mod.indices_for_weeks(arrays.start_week, test_weeks)

    seq_mean, seq_std, cal_mean, cal_std = rank_mod.fit_scaler(
        arrays.seq_x,
        arrays.future_cal_x,
        train_idx,
    )
    seq_x_scaled = rank_mod.apply_scaler(arrays.seq_x, seq_mean, seq_std)
    future_cal_scaled = rank_mod.apply_scaler(arrays.future_cal_x, cal_mean, cal_std)

    search_col_idx = [
        rank_mod.SEQUENCE_FEATURE_COLS.index(col)
        for col in rank_mod.SEARCH_FEATURE_NAMES
    ]
    feature_seq_x = seq_x_scaled.copy()

    if feature_mode == "calendar_only":
        feature_seq_x[:, :, search_col_idx] = 0.0
    elif feature_mode == "calendar_search" and not has_search:
        raise RuntimeError(
            "calendar_search mode expected, but cached search features are unavailable."
        )

    rise_target, _threshold_frame = rank_mod.build_main_targets(arrays, train_idx, quantile)

    data_arrays = rank_mod.DatasetArrays(
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

    train_dataset = rank_mod.RiseDataset(data_arrays, train_idx)
    val_dataset = rank_mod.RiseDataset(data_arrays, val_idx)
    test_dataset = rank_mod.RiseDataset(data_arrays, test_idx)

    main_pos_w, main_neg_w = rank_mod.compute_binary_class_weights(rise_target[train_idx])
    rank_pos_w, rank_neg_w = rank_mod.compute_binary_class_weights(arrays.rank_up[train_idx])
    step_pos_w, step_neg_w = rank_mod.compute_step_class_weights(arrays.step_up[train_idx])

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=rank_mod.WeekGroupedBatchSampler(
            arrays.start_week_id[train_idx],
            weeks_per_batch=rank_mod.WEEKS_PER_BATCH,
            shuffle=True,
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=rank_mod.WeekGroupedBatchSampler(
            arrays.start_week_id[val_idx],
            weeks_per_batch=rank_mod.WEEKS_PER_BATCH,
            shuffle=False,
        ),
    )

    model = rank_mod.CategoryConditionedBiGRU(
        input_dim=len(rank_mod.SEQUENCE_FEATURE_COLS),
        future_cal_dim=len(rank_mod.FUTURE_CAL_COLS),
        category_count=len(category_to_idx),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=rank_mod.LR,
        weight_decay=rank_mod.WEIGHT_DECAY,
    )

    best_state = None
    best_val_loss = float("inf")
    patience_left = rank_mod.PATIENCE

    for epoch in range(1, rank_mod.MAX_EPOCHS + 1):
        train_loss = rank_mod.run_epoch(
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
        val_loss = rank_mod.run_epoch(
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

        print(f"epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = rank_mod.PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Model failed to produce a checkpoint.")

    model.load_state_dict(best_state)

    test_rise_prob, test_rank_prob, _test_step_prob, test_score = rank_mod.collect_outputs(
        model,
        test_dataset,
        device,
    )

    test_final_score = (
        blend_weights[0] * test_score
        + blend_weights[1] * test_rise_prob
        + blend_weights[2] * test_rank_prob
    )

    test_metrics = rank_mod.compute_binary_metrics(
        rise_target[test_idx].astype(int),
        test_rise_prob,
        fixed_threshold,
    )
    test_rank_metrics = rank_mod.compute_group_ranking_metrics(
        arrays.start_week[test_idx],
        arrays.category_name[test_idx],
        rise_target[test_idx].astype(int),
        arrays.main_log_growth[test_idx],
        test_final_score,
        top_n=rank_mod.TOP_N,
    )

    final_model = rank_mod.CategoryConditionedBiGRU(
        input_dim=len(rank_mod.SEQUENCE_FEATURE_COLS),
        future_cal_dim=len(rank_mod.FUTURE_CAL_COLS),
        category_count=len(category_to_idx),
    ).to(device)
    final_model.load_state_dict(best_state)

    latest_week = dense["week_date"].max()
    future_cal = rank_mod.future_calendar_for_next_4_weeks(calendar_df, latest_week)
    future_cal_scaled = rank_mod.apply_scaler(future_cal[None, :], cal_mean, cal_std)

    future_rows = []
    future_category_scores = []
    w_score, w_rise, w_rank = blend_weights

    for category, idx in category_to_idx.items():
        group = dense.loc[dense["category"] == category].sort_values("week_date").copy()
        seq = group[rank_mod.SEQUENCE_FEATURE_COLS].to_numpy(dtype="float32")[-rank_mod.SEQ_LEN :]
        seq_scaled = rank_mod.apply_scaler(seq[None, :, :], seq_mean, seq_std)

        if feature_mode == "calendar_only":
            seq_scaled[:, :, search_col_idx] = 0.0

        seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32, device=device)
        cal_tensor = torch.tensor(future_cal_scaled, dtype=torch.float32, device=device)
        cat_tensor = torch.tensor([idx], dtype=torch.long, device=device)

        with torch.no_grad():
            rise_logit, rank_logit, step_logit, score_logit = final_model(
                seq_tensor,
                cal_tensor,
                cat_tensor,
            )
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
            }
        )

        future_category_scores.append(
            {
                "category": category,
                "rise_probability": rise_prob,
                "rank_up_probability": rank_prob,
                "ranking_score": ranking_score,
                "final_score": w_score * ranking_score + w_rise * rise_prob + w_rank * rank_prob,
            }
        )

    future_df = (
        pd.DataFrame(future_rows)
        .sort_values("ranking_score", ascending=False)
        .reset_index(drop=True)
    )
    top_df = (
        pd.DataFrame(future_category_scores)
        .sort_values("final_score", ascending=False)
        .reset_index(drop=True)
    )

    future_df.to_csv(FUTURE_PATH, index=False, encoding="utf-8-sig")
    top_df.to_csv(TOP_PATH, index=False, encoding="utf-8-sig")

    METRIC_PATH.write_text(
        json.dumps(
            {
                "feature_mode": feature_mode,
                "quantile": quantile,
                "threshold": fixed_threshold,
                "blend_weights": {
                    "ranking_score": blend_weights[0],
                    "rise_probability": blend_weights[1],
                    "rank_up_probability": blend_weights[2],
                },
                "test_metrics": test_metrics,
                "test_ranking_metrics": test_rank_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n[Top predictions]")
    print(top_df.head(10).to_string(index=False))

    print("\n[Test metrics]")
    for key, value in test_metrics.items():
        print(f"{key:<20}: {value:.3f}")

    print("\n[Test ranking metrics]")
    for key, value in test_rank_metrics.items():
        print(f"{key:<20}: {value:.3f}")

    print(f"\nsaved: {TOP_PATH}")
    print(f"saved: {FUTURE_PATH}")
    print(f"saved: {METRIC_PATH}")


if __name__ == "__main__":
    main()
