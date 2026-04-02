import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression


def create_nmme_ensemble_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics across NMME forecast models."""
    df = df.copy()

    nmme_34w_cols = [c for c in df.columns if c.startswith("nmme-tmp2m-34w__")]
    if nmme_34w_cols:
        df["nmme_34w_mean"] = df[nmme_34w_cols].mean(axis=1)
        df["nmme_34w_std"] = df[nmme_34w_cols].std(axis=1)
        df["nmme_34w_max"] = df[nmme_34w_cols].max(axis=1)
        df["nmme_34w_min"] = df[nmme_34w_cols].min(axis=1)
        df["nmme_34w_range"] = df["nmme_34w_max"] - df["nmme_34w_min"]

    nmme_56w_cols = [c for c in df.columns if c.startswith("nmme-tmp2m-56w__")]
    if nmme_56w_cols:
        df["nmme_56w_mean"] = df[nmme_56w_cols].mean(axis=1)
        df["nmme_56w_std"] = df[nmme_56w_cols].std(axis=1)

    nmme0_cols = [c for c in df.columns if c.startswith("nmme0-tmp2m-34w__")]
    if nmme0_cols:
        df["nmme0_mean"] = df[nmme0_cols].mean(axis=1)
        df["nmme0_std"] = df[nmme0_cols].std(axis=1)

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features between key variable groups."""
    df = df.copy()

    if "elevation__elevation" in df.columns and "nmme_34w_mean" in df.columns:
        df["elev_x_nmme"] = df["elevation__elevation"] * df["nmme_34w_mean"]

    if "nmme_34w_std" in df.columns and "nmme_34w_mean" in df.columns:
        df["nmme_uncertainty"] = df["nmme_34w_std"] * df["nmme_34w_mean"].abs()

    precip_cols = [
        c for c in df.columns if "precip" in c.lower() or "prate" in c.lower()
    ]
    rhum_cols = [c for c in df.columns if "rhum" in c.lower()]
    if precip_cols and rhum_cols:
        df["precip_x_rhum"] = df[precip_cols[0]] * df[rhum_cols[0]]

    return df


def select_features(
    df: pd.DataFrame,
    target_col: str = "contest-tmp2m-14d__tmp2m",
    top_k: int = 150,
) -> list[str]:
    """Select top features using mutual information with target."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {"index", target_col}
    feature_cols = [
        c for c in numeric_cols if c not in drop_cols and df[c].notna().any()
    ]

    if target_col not in df.columns:
        return feature_cols

    X = df[feature_cols].fillna(0)
    y = df[target_col]

    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_df = pd.DataFrame({"feature": feature_cols, "mi_score": mi_scores})
    mi_df = mi_df.sort_values("mi_score", ascending=False)

    selected = mi_df.head(top_k)["feature"].tolist()
    return selected


def build_features(
    df: pd.DataFrame, target_col: str = "contest-tmp2m-14d__tmp2m"
) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = create_nmme_ensemble_features(df)
    df = create_interaction_features(df)
    return df
