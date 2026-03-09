from __future__ import annotations

import argparse
import json
import math
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

matplotlib.use("Agg")

M5_DATASET_URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
DATA_FILES = ("calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv")
ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CALENDAR_COLUMNS = [
    "date",
    "wm_yr_wk",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]
CALENDAR_DTYPES = {
    "wm_yr_wk": "int16",
    "snap_CA": "int8",
    "snap_TX": "int8",
    "snap_WI": "int8",
    "event_name_1": "string",
    "event_type_1": "string",
    "event_name_2": "string",
    "event_type_2": "string",
}
PRICE_DTYPES = {
    "store_id": "string",
    "item_id": "string",
    "wm_yr_wk": "int16",
    "sell_price": "float32",
}
SNAP_COLUMNS = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
PRICE_MODEL_FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_28",
    "dow",
    "weekofyear",
    "month",
    "quarter",
    "day_of_month",
    "is_weekend",
    "snap",
    "is_event",
    "days_from_start",
    "sell_price",
    "price_ratio_median",
    "price_change_7",
]
BASE_MODEL_FEATURES = [name for name in PRICE_MODEL_FEATURES if name not in {"sell_price", "price_ratio_median", "price_change_7"}]
METHOD_LABELS = {
    "pred_no_price": "без будущей цены",
    "pred_with_price": "с будущей ценой в модели",
    "pred_curve_adjusted": "модель + кривая эластичности",
}
METHOD_CODES = {
    "pred_no_price": "no_future_price",
    "pred_with_price": "with_future_price",
    "pred_curve_adjusted": "base_plus_curve",
}
OUTPUT_FILES = {
    "selected_skus": "selected_skus.csv",
    "metrics": "forecast_metrics_by_sku.csv",
    "method_summary": "forecast_method_summary.csv",
    "elasticity_summary": "elasticity_summary.csv",
    "analysis_summary": "final_recommendations.csv",
    "predictions": "holdout_predictions.csv",
}
MODEL_PARAMS = {
    "loss": "poisson",
    "max_depth": 4,
    "learning_rate": 0.05,
    "max_iter": 250,
    "min_samples_leaf": 12,
    "l2_regularization": 0.1,
    "early_stopping": False,
}
HOLDOUT_DAYS = 28


@dataclass(slots=True)
class PipelineConfig:
    project_dir: Path
    raw_dir: Path
    processed_dir: Path
    figures_dir: Path
    tables_dir: Path
    dataset_url: str = M5_DATASET_URL
    holdout_days: int = HOLDOUT_DAYS
    selected_skus: int = 5
    min_unique_prices: int = 11
    random_state: int = 42


def default_config(project_dir: Path | None = None) -> PipelineConfig:
    root = project_dir or Path(__file__).resolve().parents[2]
    return PipelineConfig(
        project_dir=root,
        raw_dir=root / "data" / "raw",
        processed_dir=root / "data" / "processed",
        figures_dir=root / "output" / "figures",
        tables_dir=root / "output" / "tables",
    )


def demand_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("d_")]


def ensure_directories(config: PipelineConfig) -> None:
    for path in (config.raw_dir, config.processed_dir, config.figures_dir, config.tables_dir):
        path.mkdir(parents=True, exist_ok=True)


def clear_globs(path: Path, patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        for file in path.glob(pattern):
            file.unlink()


def reset_outputs(config: PipelineConfig) -> None:
    ensure_directories(config)
    clear_globs(config.figures_dir, ("*.png",))
    clear_globs(config.tables_dir, ("*.csv", "*.json"))


def download_m5_data(config: PipelineConfig, force: bool = False) -> None:
    if not force and all((config.raw_dir / name).exists() for name in DATA_FILES):
        return
    zip_path = config.raw_dir / "m5.zip"
    if force or not zip_path.exists():
        urllib.request.urlretrieve(config.dataset_url, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        members = {Path(name).name: name for name in archive.namelist()}
        for name in DATA_FILES:
            member = members.get(name)
            if member is None:
                raise FileNotFoundError(f"{name} was not found inside {zip_path.name}")
            extracted = Path(archive.extract(member, config.raw_dir))
            target = config.raw_dir / name
            if extracted == target:
                continue
            extracted.replace(target)
            parent = extracted.parent
            while parent != config.raw_dir and parent.exists():
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent


def load_sales_frame(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    dtypes = {column: "int16" for column in demand_columns(header)} | {
        "item_id": "string",
        "dept_id": "string",
        "cat_id": "string",
        "store_id": "string",
        "state_id": "string",
    }
    if "id" in header.columns:
        dtypes["id"] = "string"
    sales = pd.read_csv(path, dtype=dtypes)
    if "id" in sales.columns:
        return sales
    synthetic_id = (sales["item_id"] + "_" + sales["store_id"] + "_evaluation").rename("id")
    return pd.concat([synthetic_id, sales], axis=1)


def load_tables(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calendar = pd.read_csv(
        config.raw_dir / "calendar.csv",
        usecols=CALENDAR_COLUMNS,
        parse_dates=["date"],
        dtype=CALENDAR_DTYPES,
    ).assign(d=lambda df: "d_" + pd.Series(np.arange(1, len(df) + 1), index=df.index).astype("string"))
    sell_prices = pd.read_csv(config.raw_dir / "sell_prices.csv", dtype=PRICE_DTYPES)
    sales = load_sales_frame(config.raw_dir / "sales_train_evaluation.csv")
    return calendar, sell_prices, sales


def select_candidate_skus(sales: pd.DataFrame, sell_prices: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    demands = sales[demand_columns(sales)]
    sku_stats = sales[ID_COLUMNS].assign(
        positive_sales_ratio=demands.gt(0).mean(axis=1),
        total_sales=demands.sum(axis=1),
    )
    price_stats = sell_prices.groupby(["store_id", "item_id"], as_index=False).agg(
        unique_prices=("sell_price", "nunique"),
        min_price=("sell_price", "min"),
        median_price=("sell_price", "median"),
        max_price=("sell_price", "max"),
    )
    candidates = sku_stats.merge(price_stats, on=["store_id", "item_id"], how="inner")
    candidates = candidates.loc[candidates["unique_prices"] >= config.min_unique_prices].copy()
    if candidates.empty:
        raise RuntimeError("No item-store series satisfy the minimum unique price threshold.")
    ordered = candidates.assign(
        score=lambda df: df["positive_sales_ratio"].rank(pct=True) * 0.7 + df["total_sales"].rank(pct=True) * 0.3
    ).sort_values(["score", "positive_sales_ratio", "total_sales"], ascending=False)
    selected = ordered.drop_duplicates("item_id")
    if len(selected) < config.selected_skus:
        selected = pd.concat([selected, ordered.loc[~ordered.index.isin(selected.index)]])
    return (
        selected.head(config.selected_skus)
        .assign(sku_id=lambda df: df["store_id"] + "|" + df["item_id"])
        .reset_index(drop=True)
    )


def trim_to_available_period(series: pd.DataFrame) -> pd.DataFrame:
    valid = np.flatnonzero(series["sell_price"].notna().to_numpy())
    if len(valid) == 0:
        return series.iloc[0:0].copy()
    return series.iloc[valid[0] : valid[-1] + 1].copy().assign(sell_price=lambda df: df["sell_price"].ffill().bfill())


def build_daily_panel(
    selected_skus: pd.DataFrame,
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    sell_prices: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    meta = selected_skus.assign(sku_id=lambda df: df["store_id"] + "|" + df["item_id"])
    panel = (
        selected_skus[["store_id", "item_id"]]
        .drop_duplicates()
        .merge(sales, on=["store_id", "item_id"], how="left", validate="one_to_one")
        .melt(id_vars=ID_COLUMNS, value_vars=demand_columns(sales), var_name="d", value_name="sales")
        .merge(calendar, on="d", how="left", validate="many_to_one")
        .merge(sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one")
        .assign(
            sku_id=lambda df: df["store_id"] + "|" + df["item_id"],
            is_event=lambda df: df[["event_name_1", "event_name_2"]].notna().any(axis=1).astype("int8"),
        )
        .sort_values(["sku_id", "date"])
        .reset_index(drop=True)
    )
    panel["snap"] = np.select(
        [panel["state_id"].eq(state) for state in SNAP_COLUMNS],
        [panel[column] for column in SNAP_COLUMNS.values()],
        default=0,
    ).astype("int8")
    panel = pd.concat((trim_to_available_period(frame) for _, frame in panel.groupby("sku_id", sort=False)), ignore_index=True)
    split_day = panel["date"].max() - pd.Timedelta(days=config.holdout_days)
    panel = panel.assign(
        days_from_start=panel.groupby("sku_id").cumcount(),
        split=np.where(panel["date"] <= split_day, "train", "holdout"),
    ).merge(
        meta[["sku_id", "positive_sales_ratio", "total_sales", "unique_prices", "min_price", "median_price", "max_price"]],
        on="sku_id",
        how="left",
    )
    panel.to_csv(config.processed_dir / "selected_panel.csv", index=False)
    return panel


def add_time_features(series: pd.DataFrame, sales_column: str) -> pd.DataFrame:
    series = series.sort_values("date").reset_index(drop=True)
    demand = series[sales_column]
    price = series["sell_price"]
    return series.assign(
        **{f"lag_{lag}": demand.shift(lag) for lag in (1, 7, 14, 28)},
        rolling_mean_7=demand.shift(1).rolling(7, min_periods=7).mean(),
        rolling_mean_28=demand.shift(1).rolling(28, min_periods=28).mean(),
        rolling_std_28=demand.shift(1).rolling(28, min_periods=28).std(),
        dow=series["date"].dt.dayofweek,
        weekofyear=series["date"].dt.isocalendar().week.astype("int16"),
        month=series["date"].dt.month.astype("int8"),
        quarter=series["date"].dt.quarter.astype("int8"),
        day_of_month=series["date"].dt.day.astype("int8"),
        is_weekend=(series["date"].dt.dayofweek >= 5).astype("int8"),
        price_ratio_median=price / float(price.median()),
        price_change_7=price.pct_change(7).replace([np.inf, -np.inf], np.nan),
    )


def train_regressor(train_frame: pd.DataFrame, feature_columns: list[str], random_state: int) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(random_state=random_state, **MODEL_PARAMS)
    model.fit(train_frame[feature_columns], train_frame["sales"])
    return model


def recursive_forecast(series: pd.DataFrame, model: HistGradientBoostingRegressor, feature_columns: list[str]) -> pd.Series:
    working = series.sort_values("date").reset_index(drop=True).assign(sales_for_features=lambda df: df["sales"].astype("float64"))
    predictions: list[tuple[pd.Timestamp, float]] = []
    for index in working.index[working["split"].eq("holdout")]:
        row = add_time_features(working, "sales_for_features").loc[[index], feature_columns]
        if row.isna().any().any():
            date = working.at[index, "date"]
            raise ValueError(f"Feature NaN encountered for {working.at[index, 'sku_id']} on {date:%Y-%m-%d}")
        prediction = max(float(model.predict(row)[0]), 0.0)
        working.at[index, "sales_for_features"] = prediction
        predictions.append((working.at[index, "date"], prediction))
    return pd.Series(dict(predictions), name="prediction", dtype="float64")


def fit_elasticity_curve(
    train_frame: pd.DataFrame,
    base_model: HistGradientBoostingRegressor,
    feature_columns: list[str],
) -> dict[str, float | str | IsotonicRegression]:
    curve = train_frame.assign(
        base_prediction=np.maximum(base_model.predict(train_frame[feature_columns]), 0.5),
        price_ratio=lambda df: df["price_ratio_median"],
    )
    curve["demand_multiplier"] = (curve["sales"] / curve["base_prediction"]).clip(lower=0.05, upper=8.0)
    aggregated = curve.groupby("sell_price", as_index=False).agg(
        price_ratio=("price_ratio", "mean"),
        avg_multiplier=("demand_multiplier", "mean"),
        observations=("demand_multiplier", "size"),
    ).sort_values("price_ratio")
    if len(aggregated) < 3:
        raise RuntimeError("Not enough price points to fit the elasticity curve.")
    model = IsotonicRegression(increasing=False, out_of_bounds="clip")
    model.fit(
        aggregated["price_ratio"].to_numpy(),
        aggregated["avg_multiplier"].to_numpy(),
        sample_weight=aggregated["observations"].to_numpy(),
    )
    bounds = aggregated["price_ratio"].agg(["min", "max"])
    lower_ratio, upper_ratio = [
        float(np.clip(np.quantile(curve["price_ratio"], quantile), bounds["min"], bounds["max"]))
        for quantile in (0.10, 0.90)
    ]
    reference_multiplier = float(model.predict([1.0])[0])
    lower_multiplier, upper_multiplier = (float(model.predict([ratio])[0]) for ratio in (lower_ratio, upper_ratio))
    price_span = math.log(max(upper_ratio, 1e-6)) - math.log(max(lower_ratio, 1e-6))
    demand_span = math.log(max(upper_multiplier, 1e-6)) - math.log(max(lower_multiplier, 1e-6))
    local_elasticity = 0.0 if abs(price_span) < 1e-12 else float(demand_span / price_span)
    promo_lift = float(lower_multiplier / reference_multiplier - 1.0)
    return {
        "model": model,
        "reference_multiplier": reference_multiplier,
        "local_elasticity": local_elasticity,
        "promo_lift": promo_lift,
        "recommendation": "promo" if local_elasticity <= -1.0 and promo_lift >= 0.15 else "regular",
    }


def score_forecast(actual: pd.Series, prediction: pd.Series) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "wape": float(np.abs(actual - prediction).sum() / np.maximum(actual.sum(), 1.0)),
        "bias": float((prediction - actual).mean()),
    }


def figure_path(figures_dir: Path, prefix: str, sku_id: str) -> Path:
    return figures_dir / f"{prefix}_{sku_id.replace('|', '_')}.png"


def save_elasticity_plot(
    series: pd.DataFrame,
    curve: dict[str, float | str | IsotonicRegression],
    sku_row: pd.Series,
    figures_dir: Path,
) -> None:
    model = curve["model"]
    if not isinstance(model, IsotonicRegression):
        raise TypeError("Unexpected elasticity model type.")
    grouped = series.groupby("sell_price", as_index=False).agg(
        price_ratio_median=("price_ratio_median", "mean"),
        sales=("sales", "mean"),
    ).sort_values("price_ratio_median")
    ratio_grid = np.linspace(series["price_ratio_median"].min(), series["price_ratio_median"].max(), 200)
    scaled_curve = model.predict(ratio_grid) / float(curve["reference_multiplier"]) * float(series["sales"].median())
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.scatter(grouped["price_ratio_median"], grouped["sales"], alpha=0.7, label="avg sales by price")
    axis.plot(ratio_grid, scaled_curve, color="#b22222", linewidth=2.5, label="elasticity curve (scaled)")
    axis.axvline(1.0, color="#444444", linestyle="--", linewidth=1, label="median price")
    axis.set(title=f"Elasticity curve for {sku_row['sku_id']}", xlabel="Price / median price", ylabel="Average sales (scaled)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path(figures_dir, "elasticity", sku_row["sku_id"]), dpi=160)
    plt.close(figure)


def save_forecast_plot(forecast_frame: pd.DataFrame, sku_row: pd.Series, figures_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(forecast_frame["date"], forecast_frame["sales"], color="#111111", linewidth=2.0, label="actual")
    axis.plot(forecast_frame["date"], forecast_frame["pred_no_price"], color="#1f77b4", linewidth=1.8, label="no future price")
    axis.plot(forecast_frame["date"], forecast_frame["pred_with_price"], color="#2ca02c", linewidth=1.8, label="with future price")
    axis.plot(
        forecast_frame["date"],
        forecast_frame["pred_curve_adjusted"],
        color="#d62728",
        linewidth=1.8,
        label="model + elasticity",
    )
    axis.set(title=f"Holdout forecast for {sku_row['sku_id']}", xlabel="Date", ylabel="Daily sales")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path(figures_dir, "forecast", sku_row["sku_id"]), dpi=160)
    plt.close(figure)


def summarize_item_result(
    sku_row: pd.Series,
    metrics_frame: pd.DataFrame,
    elasticity: dict[str, float | str | IsotonicRegression],
) -> str:
    best = metrics_frame.sort_values("wape").iloc[0]
    return (
        f"{sku_row['sku_id']}: лучшая WAPE у подхода '{METHOD_LABELS[best['method']]}' ({best['wape']:.3f}). "
        f"Локальная эластичность {float(elasticity['local_elasticity']):.2f}, "
        f"ожидаемый lift на промо {float(elasticity['promo_lift']) * 100:.1f}%, "
        f"рекомендация: {elasticity['recommendation']}."
    )


def analyze_single_sku(series: pd.DataFrame, sku_row: pd.Series, config: PipelineConfig) -> dict[str, pd.DataFrame | str]:
    series = series.sort_values("date").reset_index(drop=True)
    features = add_time_features(series, sales_column="sales")
    train = features.loc[features["split"].eq("train")].dropna(subset=PRICE_MODEL_FEATURES).copy()
    if len(train) < 365:
        raise RuntimeError(f"Too little training history for {sku_row['sku_id']}")
    base_model = train_regressor(train, BASE_MODEL_FEATURES, config.random_state)
    price_model = train_regressor(train, PRICE_MODEL_FEATURES, config.random_state)
    predictions = pd.concat(
        [
            recursive_forecast(series, base_model, BASE_MODEL_FEATURES).rename("pred_no_price"),
            recursive_forecast(series, price_model, PRICE_MODEL_FEATURES).rename("pred_with_price"),
        ],
        axis=1,
    ).reset_index(names="date")
    elasticity = fit_elasticity_curve(train, base_model, BASE_MODEL_FEATURES)
    model = elasticity["model"]
    if not isinstance(model, IsotonicRegression):
        raise TypeError("Unexpected elasticity model type.")
    forecast = features.loc[features["split"].eq("holdout"), ["date", "sales", "sell_price", "price_ratio_median"]].merge(
        predictions,
        on="date",
        how="left",
    )
    forecast["pred_curve_adjusted"] = (
        forecast["pred_no_price"].to_numpy()
        * model.predict(forecast["price_ratio_median"]) / float(elasticity["reference_multiplier"])
    ).clip(min=0.0)
    indexed = forecast.set_index("date")
    metrics = pd.DataFrame(
        [
            {"sku_id": sku_row["sku_id"], "method": method, **score_forecast(indexed["sales"], indexed[method])}
            for method in METHOD_LABELS
        ]
    )
    elasticity_frame = pd.DataFrame(
        [
            {
                "sku_id": sku_row["sku_id"],
                "item_id": sku_row["item_id"],
                "store_id": sku_row["store_id"],
                "median_price": sku_row["median_price"],
                "unique_prices": sku_row["unique_prices"],
                "local_elasticity": float(elasticity["local_elasticity"]),
                "promo_lift": float(elasticity["promo_lift"]),
                "recommendation": str(elasticity["recommendation"]),
            }
        ]
    )
    save_elasticity_plot(train, elasticity, sku_row, config.figures_dir)
    save_forecast_plot(forecast, sku_row, config.figures_dir)
    return {
        "forecast": forecast.assign(sku_id=sku_row["sku_id"]).loc[
            :,
            ["sku_id", "date", "sales", "sell_price", "pred_no_price", "pred_with_price", "pred_curve_adjusted"],
        ],
        "metrics": metrics,
        "elasticity": elasticity_frame,
        "narrative": summarize_item_result(sku_row, metrics, elasticity),
    }


def write_outputs(config: PipelineConfig, frames: dict[str, pd.DataFrame], narratives: list[str]) -> None:
    for name, frame in frames.items():
        frame.to_csv(config.tables_dir / OUTPUT_FILES[name], index=False)
    summary = {
        "selected_skus": len(frames["selected_skus"]),
        "best_method_by_avg_wape": frames["method_summary"].iloc[0]["method"],
        "mean_wape": float(frames["method_summary"].iloc[0]["wape"]),
        "narratives": narratives,
    }
    (config.tables_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pipeline(config: PipelineConfig, force_download: bool = False) -> dict[str, pd.DataFrame | list[str]]:
    reset_outputs(config)
    download_m5_data(config, force=force_download)
    calendar, sell_prices, sales = load_tables(config)
    selected_skus = select_candidate_skus(sales, sell_prices, config)
    panel = build_daily_panel(selected_skus, sales, calendar, sell_prices, config)
    results = [
        analyze_single_sku(panel.loc[panel["sku_id"].eq(sku_row["sku_id"])].copy(), sku_row, config)
        for _, sku_row in selected_skus.iterrows()
    ]
    metrics = pd.concat([result["metrics"] for result in results], ignore_index=True)
    elasticity_summary = pd.concat([result["elasticity"] for result in results], ignore_index=True)
    predictions = pd.concat([result["forecast"] for result in results], ignore_index=True)
    narratives = [str(result["narrative"]) for result in results]
    method_summary = metrics.groupby("method", as_index=False).agg(
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        wape=("wape", "mean"),
        bias=("bias", "mean"),
    ).sort_values("wape").reset_index(drop=True)
    best_by_sku = metrics.sort_values("wape").drop_duplicates("sku_id")[["sku_id", "method", "wape"]]
    analysis_summary = elasticity_summary.merge(best_by_sku, on="sku_id", how="left").assign(
        best_method=lambda df: df["method"].map(METHOD_CODES)
    ).drop(columns="method")
    frames = {
        "selected_skus": selected_skus,
        "metrics": metrics,
        "method_summary": method_summary,
        "elasticity_summary": elasticity_summary,
        "analysis_summary": analysis_summary,
        "predictions": predictions,
    }
    write_outputs(config, frames, narratives)
    return {**frames, "narratives": narratives}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HW3 M5 elasticity and forecasting analysis.")
    parser.add_argument("--force-download", action="store_true", help="Re-download and re-extract the M5 dataset.")
    parser.add_argument("--selected-skus", type=int, default=5, help="Number of item-store series to analyze.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid")
    config = replace(default_config(), selected_skus=args.selected_skus)
    results = run_pipeline(config, force_download=args.force_download)
    print(
        json.dumps(
            {
                "selected_skus": results["selected_skus"]["sku_id"].tolist(),
                "method_summary": results["method_summary"].to_dict(orient="records"),
                "narratives": results["narratives"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
