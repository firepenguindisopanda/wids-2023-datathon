import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.preprocess import load_data, preprocess
from src.features import build_features, select_features
from src.evaluate import compute_metrics, compare_to_baseline


def export_feature_importance(model, feature_names, output_path: str):
    """Export top 20 feature importances as JSON."""
    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[
        :20
    ]

    data = {
        "features": [
            {"name": name, "importance": round(float(imp), 4)} for name, imp in feat_imp
        ],
        "total_features": 20,
        "model_type": "LightGBM",
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported feature importance to {output_path}")


def export_model_metrics(comparison_report, cv_results, n_samples, output_path: str):
    """Export model metrics and baseline comparison as JSON."""
    data = {
        "lightgbm": comparison_report["model"],
        "baseline_nmme_mean": comparison_report["baseline_nmme_mean"],
        "improvement": comparison_report["improvement"],
        "cv_folds": 5,
        "training_samples": n_samples,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported model metrics to {output_path}")


def export_predictions_sample(y_true, y_pred, df, output_path: str, n: int = 200):
    """Export stratified sample of predictions vs actuals as JSON."""
    residuals = pd.DataFrame(
        {
            "actual": y_true,
            "predicted": y_pred,
            "error": (y_true - y_pred).round(4),
        }
    )

    for col in ["month", "season", "startdate", "climateregions__climateregion"]:
        if col in df.columns:
            residuals[col] = df[col].values

    if "season" in residuals.columns:
        sample = residuals.groupby("season", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), n // 4), random_state=42)
        )
    else:
        sample = residuals.sample(n=min(n, len(residuals)), random_state=42)

    samples = []
    for _, row in sample.iterrows():
        entry = {
            "actual": round(float(row["actual"]), 4),
            "predicted": round(float(row["predicted"]), 4),
            "error": round(float(row["error"]), 4),
        }
        if "month" in row:
            entry["month"] = int(row["month"])
        if "season" in row:
            entry["season"] = int(row["season"])
        if "startdate" in row:
            entry["startdate"] = str(row["startdate"])
        if "climateregions__climateregion" in row:
            entry["climate_region"] = str(row["climateregions__climateregion"])
        samples.append(entry)

    data = {
        "samples": samples,
        "sample_size": len(samples),
        "sampling_method": "stratified_by_season",
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported predictions sample to {output_path}")


def export_eda_stats(train, output_path: str):
    """Export EDA summary statistics as JSON."""
    target_col = "contest-tmp2m-14d__tmp2m"

    data = {
        "dataset": {
            "train_rows": int(len(train)),
            "total_features": int(train.shape[1] - 4),
            "date_range": f"{train['startdate'].min()} to {train['startdate'].max()}",
            "unique_locations": int(train[["lat", "lon"]].drop_duplicates().shape[0]),
        },
        "target": {
            "mean": round(float(train[target_col].mean()), 2),
            "std": round(float(train[target_col].std()), 2),
            "min": round(float(train[target_col].min()), 2),
            "max": round(float(train[target_col].max()), 2),
            "median": round(float(train[target_col].median()), 2),
        },
        "missing_values": {
            "total_missing": int(train.isnull().sum().sum()),
            "features_with_missing": int((train.isnull().sum() > 0).sum()),
            "max_missing_pct": round(float(train.isnull().mean().max() * 100), 2),
        },
        "climate_regions": train["climateregions__climateregion"].unique().tolist()
        if "climateregions__climateregion" in train.columns
        else [],
        "seasonal_patterns": {},
    }

    if "season" in train.columns:
        season_labels = {0: "winter", 1: "spring", 2: "summer", 3: "fall"}
        seasonal = train.groupby("season")[target_col].mean()
        data["seasonal_patterns"] = {
            season_labels.get(k, str(k)): round(float(v), 1)
            for k, v in seasonal.items()
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported EDA stats to {output_path}")


def generate_all_plots(model, X, y_true, y_pred, train_df, output_dir: str):
    """Generate all static plots for portfolio."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    # 1. Feature importance horizontal bar
    importance = model.feature_importances_
    feat_imp = (
        pd.DataFrame({"feature": X.columns, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(20)
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(feat_imp["feature"], feat_imp["importance"])
    ax.invert_yaxis()
    ax.set_title("LightGBM Feature Importance (Top 20)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Generated feature_importance.png")

    # 2. Prediction vs Actual scatter
    fig, ax = plt.subplots(figsize=(10, 10))
    sample_idx = np.random.choice(len(y_true), size=5000, replace=False)
    ax.scatter(
        y_true.iloc[sample_idx], y_pred[sample_idx], alpha=0.3, s=10, edgecolors="none"
    )
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=2, label="Perfect prediction")
    ax.set_xlabel("Actual Temperature (°C)", fontsize=12)
    ax.set_ylabel("Predicted Temperature (°C)", fontsize=12)
    ax.set_title("Predicted vs Actual Temperature", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "prediction_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Generated prediction_vs_actual.png")

    # 3. Residual plot
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, edgecolors="none")
    axes[0].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[0].set_xlabel("Predicted", fontsize=12)
    axes[0].set_ylabel("Residual", fontsize=12)
    axes[0].set_title("Residuals vs Predicted", fontsize=14, fontweight="bold")
    sns.histplot(residuals, bins=100, kde=True, ax=axes[1])
    axes[1].set_title("Residual Distribution", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Residual", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / "model_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Generated model_residuals.png")

    # 4. Target distribution
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(y_true, bins=100, kde=True, ax=ax)
    ax.set_title("Target Variable Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Temperature (°C)")
    plt.tight_layout()
    fig.savefig(output_dir / "target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Generated target_distribution.png")

    # 5. Correlation heatmap (top 30 features)
    corr_cols = X.columns[:30].tolist() + [y_true.name]
    corr = train_df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr[[y_true.name]].sort_values(y_true.name, ascending=False),
        annot=True,
        cmap="RdBu_r",
        center=0,
        ax=ax,
        fmt=".2f",
    )
    ax.set_title("Feature Correlation with Target", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Generated correlation_heatmap.png")

    # 6. Spatial temperature map
    if "lat" in train_df.columns and "lon" in train_df.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        location_means = (
            train_df.groupby(["lat", "lon"])[y_true.name].mean().reset_index()
        )
        scatter = ax.scatter(
            location_means["lon"],
            location_means["lat"],
            c=location_means[y_true.name],
            cmap="RdYlBu_r",
            s=50,
            alpha=0.7,
        )
        plt.colorbar(scatter, label="Mean Temperature (°C)")
        ax.set_title("Spatial Temperature Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()
        fig.savefig(
            output_dir / "spatial_temperature_map.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        print("Generated spatial_temperature_map.png")


def main():
    """Main export pipeline."""
    print("=" * 60)
    print("Exporting portfolio artifacts...")
    print("=" * 60)

    base = Path(__file__).parent.parent
    output_dir = base / "outputs"
    plots_dir = output_dir / "plots"
    portfolio_dir = output_dir / "portfolio_data"

    print("\n[1/6] Loading data...")
    train_raw, _ = load_data(str(base / "data"))
    train, _ = preprocess(train_raw, train_raw)
    train = build_features(train)

    target_col = "contest-tmp2m-14d__tmp2m"

    # Load pre-selected features instead of recomputing MI on full dataset
    selected_path = base / "outputs" / "portfolio_data" / "selected_features.json"
    if selected_path.exists():
        with open(selected_path) as f:
            selected = json.load(f)["selected_features"]
        print(f"Loaded {len(selected)} pre-selected features")
    else:
        selected = select_features(train, target_col, top_k=150)
        print(f"Selected {len(selected)} features")

    X = train[selected].fillna(0)
    y = train[target_col]

    print("[2/6] Loading trained model...")
    model_path = output_dir / "final_model.pkl"
    if not model_path.exists():
        print(
            "ERROR: Model file not found. Run notebooks/03_model_training.ipynb first."
        )
        return
    model = joblib.load(model_path)

    print("[3/6] Generating predictions...")
    y_pred = model.predict(X)

    print("[4/6] Computing metrics...")
    nmme_baseline_col = "nmme-tmp2m-34w__nmmemean"
    baseline_pred = train[nmme_baseline_col].values
    comparison = compare_to_baseline(y.values, y_pred, baseline_pred)

    print("[5/6] Exporting JSON artifacts...")
    export_feature_importance(
        model, selected, str(portfolio_dir / "feature_importance.json")
    )
    export_model_metrics(
        comparison, None, len(train), str(portfolio_dir / "model_metrics.json")
    )
    export_predictions_sample(
        y, y_pred, train, str(portfolio_dir / "predictions_sample.json")
    )
    export_eda_stats(train_raw, str(portfolio_dir / "eda_stats.json"))

    print("[6/6] Generating plots...")
    generate_all_plots(model, X, y, y_pred, train, str(plots_dir))

    print("\n" + "=" * 60)
    print("All artifacts exported successfully!")
    print(f"  Plots: {plots_dir}")
    print(f"  JSON data: {portfolio_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
