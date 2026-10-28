import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from segmentation_utils import (
    DEFAULT_FEATURES,
    has_no_purchase_history,
    load_customer_data,
    predict_customer_cluster,
    segment_names,
)


class LoadCustomerDataTests(unittest.TestCase):
    def write_csv(self, directory: str, data: pd.DataFrame) -> Path:
        path = Path(directory) / "customers.csv"
        data.to_csv(path, index=False)
        return path

    def valid_data(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Customer_ID": [f"C{i}" for i in range(6)],
                "Recency": [1, 2, 3, 4, 5, 6],
                "Frequency": [6, 5, 4, 3, 2, 1],
                "Monetary": [100, 200, 300, 400, 500, 600],
            }
        )

    def test_loads_valid_data_and_removes_duplicate_rows(self) -> None:
        data = self.valid_data()
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, data)
            loaded = load_customer_data(path)

        self.assertEqual(len(loaded), 6)
        self.assertTrue(all(pd.api.types.is_numeric_dtype(loaded[column]) for column in DEFAULT_FEATURES))

    def test_rejects_missing_required_columns(self) -> None:
        data = self.valid_data().drop(columns=["Monetary"])
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, data)
            with self.assertRaisesRegex(ValueError, "Monetary"):
                load_customer_data(path)

    def test_rejects_non_numeric_values(self) -> None:
        data = self.valid_data()
        data["Frequency"] = data["Frequency"].astype(object)
        data.loc[0, "Frequency"] = "unknown"
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, data)
            with self.assertRaisesRegex(ValueError, "non-numeric"):
                load_customer_data(path)

    def test_rejects_missing_feature_values(self) -> None:
        data = self.valid_data()
        data.loc[0, "Monetary"] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, data)
            with self.assertRaisesRegex(ValueError, "missing or non-numeric"):
                load_customer_data(path)

    def test_rejects_negative_values(self) -> None:
        data = self.valid_data()
        data.loc[0, "Recency"] = -1
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, data)
            with self.assertRaisesRegex(ValueError, "cannot be negative"):
                load_customer_data(path)

    def test_rejects_too_few_distinct_rfm_rows(self) -> None:
        data = self.valid_data()
        data["Recency"] = [1, 1, 2, 2, 1, 2]
        data["Frequency"] = [1, 1, 2, 2, 1, 2]
        data["Monetary"] = 100
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, data)
            with self.assertRaisesRegex(ValueError, "distinct RFM rows"):
                load_customer_data(path)

    def test_rejects_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_customer_data(Path(directory) / "missing.csv")


class SegmentationHelperTests(unittest.TestCase):
    def test_zero_frequency_and_monetary_means_no_purchase_history(self) -> None:
        self.assertTrue(has_no_purchase_history(0, 0))
        self.assertFalse(has_no_purchase_history(15, 0))
        self.assertFalse(has_no_purchase_history(0, 125))
        self.assertFalse(has_no_purchase_history(5, 125))

    def test_no_purchase_history_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            has_no_purchase_history(-1, 0)
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            has_no_purchase_history(0, np.nan)

    def test_names_clusters_by_relative_rfm_priority(self) -> None:
        profile = pd.DataFrame(
            {
                "Recency": [5, 60],
                "Frequency": [20, 2],
                "Monetary": [2000, 100],
            },
            index=[3, 8],
        )

        self.assertEqual(segment_names(profile), {3: "Champions", 8: "Loyal Customers"})

    def test_predicts_cluster_nearest_to_new_customer(self) -> None:
        training = np.array(
            [
                [1, 1, 10],
                [2, 1, 12],
                [1, 2, 11],
                [50, 20, 1000],
                [60, 25, 1200],
                [55, 22, 1100],
            ],
            dtype=float,
        )
        scaler = StandardScaler().fit(training)
        model = KMeans(n_clusters=2, random_state=42, n_init=10).fit(scaler.transform(training))
        customer = [2, 2, 13]
        predicted_cluster = predict_customer_cluster(scaler, model, customer)

        nearest_customer_index = np.argmin(
            np.linalg.norm(scaler.transform(training) - scaler.transform([customer]), axis=1)
        )
        self.assertEqual(predicted_cluster, int(model.labels_[nearest_customer_index]))

    def test_rejects_invalid_prediction_input(self) -> None:
        training = np.array([[1, 1, 10], [2, 2, 20], [3, 3, 30], [4, 4, 40], [5, 5, 50], [6, 6, 60]])
        scaler = StandardScaler().fit(training)
        model = KMeans(n_clusters=2, random_state=42, n_init=10).fit(scaler.transform(training))

        with self.assertRaisesRegex(ValueError, "Expected 3"):
            predict_customer_cluster(scaler, model, [1, 2])
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            predict_customer_cluster(scaler, model, [1, np.nan, 3])


if __name__ == "__main__":
    unittest.main()
