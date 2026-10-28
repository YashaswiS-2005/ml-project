"""Data validation and reusable helpers for the customer segmentation app."""

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = ("Recency", "Frequency", "Monetary")
MAX_CLUSTERS = 6


def load_customer_data(path: Path) -> pd.DataFrame:
    """Load the customer CSV and validate the features required by the app."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        data = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read dataset CSV: {error}") from error

    missing_columns = [column for column in DEFAULT_FEATURES if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    data = data.drop_duplicates().copy()
    for column in DEFAULT_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    invalid_counts = {
        column: int((~np.isfinite(data[column])).sum())
        for column in DEFAULT_FEATURES
    }
    invalid_counts = {column: count for column, count in invalid_counts.items() if count}
    if invalid_counts:
        details = ", ".join(f"{column}: {count}" for column, count in invalid_counts.items())
        raise ValueError(f"Dataset has missing or non-numeric RFM values ({details}).")

    negative_counts = {
        column: int((data[column] < 0).sum())
        for column in DEFAULT_FEATURES
        if (data[column] < 0).any()
    }
    if negative_counts:
        details = ", ".join(f"{column}: {count}" for column, count in negative_counts.items())
        raise ValueError(f"RFM values cannot be negative ({details}).")

    unique_feature_rows = data[list(DEFAULT_FEATURES)].drop_duplicates().shape[0]
    if unique_feature_rows < MAX_CLUSTERS:
        raise ValueError(
            f"Dataset needs at least {MAX_CLUSTERS} distinct RFM rows to support the cluster selector; "
            f"found {unique_feature_rows}."
        )

    return data


def segment_names(profile: pd.DataFrame) -> dict[int, str]:
    """Assign heuristic, presentation-friendly labels to the anonymous clusters."""
    labels = [
        "Champions",
        "Loyal Customers",
        "Potential Customers",
        "At-Risk Customers",
        "Low-Engagement Customers",
        "Other Customers",
    ]
    if len(profile) > len(labels):
        raise ValueError(f"Cannot label more than {len(labels)} clusters.")

    priority = (
        profile["Monetary"].rank(pct=True)
        + profile["Frequency"].rank(pct=True)
        - profile["Recency"].rank(pct=True)
    )
    ordered = priority.sort_values(ascending=False).index.tolist()
    return {cluster: labels[position] for position, cluster in enumerate(ordered)}


def predict_customer_cluster(
    scaler: StandardScaler,
    model: KMeans,
    values: Sequence[float],
) -> int:
    """Assign one RFM observation using the app's fitted scaler and K-Means model."""
    customer = np.asarray(values, dtype=float)
    if customer.shape != (len(DEFAULT_FEATURES),):
        raise ValueError(f"Expected {len(DEFAULT_FEATURES)} RFM values.")
    if not np.isfinite(customer).all() or (customer < 0).any():
        raise ValueError("RFM values must be finite and non-negative.")
    return int(model.predict(scaler.transform(customer.reshape(1, -1)))[0])


def has_no_purchase_history(frequency: float, monetary: float) -> bool:
    """Return whether the customer has neither recorded orders nor spend."""
    purchase_values = np.asarray([frequency, monetary], dtype=float)
    if not np.isfinite(purchase_values).all() or (purchase_values < 0).any():
        raise ValueError("Frequency and Monetary must be finite and non-negative.")
    return bool(np.all(purchase_values == 0))
