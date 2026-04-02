import pandas as pd
import numpy as np
from pathlib import Path


def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSV files."""
    data_path = Path(data_dir)
    train = pd.read_csv(data_path / "train_data.csv")
    test = pd.read_csv(data_path / "test_data.csv")
    return train, test


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse startdate strings into datetime and extract temporal components."""
    df = df.copy()
    df["startdate_dt"] = pd.to_datetime(df["startdate"], format="mixed")
    df["month"] = df["startdate_dt"].dt.month
    df["day_of_year"] = df["startdate_dt"].dt.dayofyear
    df["year"] = df["startdate_dt"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["season"] = df["month"].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )
    return df


def encode_climate_region(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode climate region column."""
    df = df.copy()
    if "climateregions__climateregion" in df.columns:
        df = pd.get_dummies(
            df, columns=["climateregions__climateregion"], prefix="climate", dtype=float
        )
    return df


def get_feature_columns(
    df: pd.DataFrame, target_col: str = "contest-tmp2m-14d__tmp2m"
) -> list[str]:
    """Get all numeric feature columns, excluding identifiers and target."""
    drop_cols = {"index", "lat", "lon", "startdate", "startdate_dt", target_col}
    return [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in drop_cols
    ]


def preprocess(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full preprocessing pipeline for both train and test."""
    train = parse_dates(train)
    test = parse_dates(test)
    train = encode_climate_region(train)
    test = encode_climate_region(test)
    return train, test
