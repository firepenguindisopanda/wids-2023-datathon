import numpy as np
import pandas as pd


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, and R²."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "r2": round(float(r2), 4),
    }


def compare_to_baseline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: np.ndarray,
) -> dict:
    """Compare model performance to baseline (NMME mean forecast)."""
    model_metrics = compute_metrics(y_true, y_pred)
    baseline_metrics = compute_metrics(y_true, baseline_pred)

    rmse_improvement = (
        (baseline_metrics["rmse"] - model_metrics["rmse"])
        / baseline_metrics["rmse"]
        * 100
    )
    mae_improvement = (
        (baseline_metrics["mae"] - model_metrics["mae"]) / baseline_metrics["mae"] * 100
    )

    return {
        "model": model_metrics,
        "baseline_nmme_mean": baseline_metrics,
        "improvement": {
            "rmse_reduction_pct": round(rmse_improvement, 2),
            "mae_reduction_pct": round(mae_improvement, 2),
        },
    }


def analyze_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create residuals DataFrame with contextual features for analysis."""
    residuals = pd.DataFrame(
        {
            "actual": y_true,
            "predicted": y_pred,
            "error": y_true - y_pred,
            "abs_error": np.abs(y_true - y_pred),
        }
    )

    for col in ["month", "season", "startdate_dt", "elevation__elevation"]:
        if col in df.columns:
            residuals[col] = df[col].values

    return residuals


def generate_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: np.ndarray,
    df: pd.DataFrame,
    cv_results: dict = None,
    n_samples: int = 375734,
) -> dict:
    """Generate comprehensive evaluation report."""
    comparison = compare_to_baseline(y_true, y_pred, baseline_pred)
    residuals = analyze_residuals(y_true, y_pred, df)

    if "season" in residuals.columns:
        error_by_season = residuals.groupby("season")["abs_error"].mean().to_dict()
    else:
        error_by_season = {}

    if "month" in residuals.columns:
        error_by_month = residuals.groupby("month")["abs_error"].mean().to_dict()
    else:
        error_by_month = {}

    report = {
        "comparison": comparison,
        "cv_results": cv_results,
        "error_by_season": {str(k): round(v, 4) for k, v in error_by_season.items()},
        "error_by_month": {str(k): round(v, 4) for k, v in error_by_month.items()},
        "training_samples": int(n_samples),
    }

    return report
