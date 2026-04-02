import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from pathlib import Path
import json


def get_default_params() -> dict:
    """Sensible default LightGBM hyperparameters for regression."""
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbose": -1,
    }


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict = None,
) -> lgb.LGBMRegressor:
    """Train LightGBM with early stopping on validation set."""
    if params is None:
        params = get_default_params()

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    return model


def cross_validate(
    X: pd.DataFrame, y: pd.Series, params: dict = None, n_folds: int = 5
) -> dict:
    """K-fold cross-validation returning per-fold and mean metrics."""
    if params is None:
        params = get_default_params()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )

        preds = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - preds) ** 2))
        mae = np.mean(np.abs(y_val - preds))
        r2 = 1 - np.sum((y_val - preds) ** 2) / np.sum((y_val - y_val.mean()) ** 2)

        fold_scores.append({"fold": fold + 1, "rmse": rmse, "mae": mae, "r2": r2})
        print(f"Fold {fold + 1}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    mean_rmse = np.mean([s["rmse"] for s in fold_scores])
    mean_mae = np.mean([s["mae"] for s in fold_scores])
    mean_r2 = np.mean([s["r2"] for s in fold_scores])

    print(f"\nMean: RMSE={mean_rmse:.4f}, MAE={mean_mae:.4f}, R²={mean_r2:.4f}")

    return {
        "folds": fold_scores,
        "mean_rmse": mean_rmse,
        "mean_mae": mean_mae,
        "mean_r2": mean_r2,
    }


def train_final_model(
    X: pd.DataFrame, y: pd.Series, params: dict = None
) -> lgb.LGBMRegressor:
    """Train final model on all available data (no early stopping validation)."""
    if params is None:
        params = get_default_params()

    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    return model


def save_model(model: lgb.LGBMRegressor, path: str = "outputs/model.txt"):
    """Save trained model to file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(path)
