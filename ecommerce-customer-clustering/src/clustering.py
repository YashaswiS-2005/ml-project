"""E-commerce Customer Behaviour Clustering

This beginner-friendly project uses RFM analysis and K-Means clustering to group customers
based on their purchasing habits. The script automatically creates a sample e-commerce
transaction dataset if no CSV file is available, then cleans the data, builds customer-level
RFM features, evaluates clustering quality, and saves the final customer segments.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "ecommerce.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_column(df: pd.DataFrame, possible_names: list[str]) -> str | None:
    """Return the matching column name if found, otherwise None."""
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def create_sample_dataset(path: Path) -> pd.DataFrame:
    """Create a realistic sample e-commerce dataset for learning purposes."""
    np.random.seed(42)
    data = []

    customer_profiles = {
        "High Value": {"customers": 45, "avg_orders": 18, "avg_value": 850},
        "Regular": {"customers": 70, "avg_orders": 10, "avg_value": 500},
        "New": {"customers": 55, "avg_orders": 5, "avg_value": 210},
        "At Risk": {"customers": 35, "avg_orders": 6, "avg_value": 180},
    }

    customer_id = 1
    end_date = pd.Timestamp("2024-01-31")

    for segment_name, profile in customer_profiles.items():
        for _ in range(profile["customers"]):
            order_count = max(2, int(np.random.normal(profile["avg_orders"], profile["avg_orders"] * 0.35)))
            order_count = min(order_count, 30)

            for _ in range(order_count):
                if segment_name == "High Value":
                    days_ago = int(np.random.randint(1, 45))
                    unit_price = round(np.random.uniform(20, 120), 2)
                    quantity = int(np.random.randint(1, 8))
                elif segment_name == "Regular":
                    days_ago = int(np.random.randint(10, 90))
                    unit_price = round(np.random.uniform(10, 80), 2)
                    quantity = int(np.random.randint(1, 7))
                elif segment_name == "New":
                    days_ago = int(np.random.randint(1, 35))
                    unit_price = round(np.random.uniform(8, 60), 2)
                    quantity = int(np.random.randint(1, 6))
                else:  # At Risk
                    days_ago = int(np.random.randint(30, 180))
                    unit_price = round(np.random.uniform(5, 55), 2)
                    quantity = int(np.random.randint(1, 5))

                invoice_date = end_date - pd.Timedelta(days=days_ago)
                invoice_no = f"INV-{customer_id}-{len(data) + 1:04d}"
                data.append(
                    {
                        "CustomerID": customer_id,
                        "InvoiceNo": invoice_no,
                        "InvoiceDate": invoice_date.strftime("%Y-%m-%d"),
                        "Quantity": quantity,
                        "UnitPrice": unit_price,
                    }
                )

            customer_id += 1

    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_and_clean_data(path: Path) -> pd.DataFrame:
    """Load the data, adapt column names, and remove invalid or duplicate rows."""
    if not path.exists():
        print(f"Dataset not found. Creating a sample dataset at {path}")
        create_sample_dataset(path)

    df = pd.read_csv(path)
    print("\nFirst few rows of the dataset:")
    print(df.head())
    print("\nDataset information:")
    df.info()

    required_columns = {
        "CustomerID": ["CustomerID", "Customer ID", "customer_id", "Customer"],
        "InvoiceNo": ["InvoiceNo", "OrderID", "Order ID", "InvoiceNo/OrderID", "invoice_no"],
        "InvoiceDate": ["InvoiceDate", "OrderDate", "Date", "invoice_date", "order_date"],
        "Quantity": ["Quantity", "Qty", "qnty"],
        "UnitPrice": ["UnitPrice", "Unit Price", "Price", "unit_price"],
    }

    renamed = {}
    for standard_name, possible_names in required_columns.items():
        actual_name = find_column(df, possible_names)
        if actual_name is not None:
            renamed[actual_name] = standard_name

    df = df.rename(columns=renamed)

    missing_before = df.isnull().sum()
    print("\nMissing values before cleaning:")
    print(missing_before)

    for col in ["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]:
        if col in df.columns:
            df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

    df = df.dropna(subset=["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]).copy()
    df = df.drop_duplicates().reset_index(drop=True)

    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    df = df.dropna(subset=["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]).copy()
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()

    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    print(f"\nCleaned dataset shape: {df.shape}")
    return df


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create one row per customer using RFM features."""
    latest_date = df["InvoiceDate"].max()

    rfm = (
        df.groupby("CustomerID", as_index=True)
        .agg(
            Recency=("InvoiceDate", lambda x: (latest_date - x.max()).days),
            Frequency=("InvoiceDate", "count"),
            Monetary=("TotalAmount", "sum"),
        )
        .reset_index()
    )

    # Use order count instead of transaction count as a cleaner frequency measure.
    order_frequency = (
        df.groupby("CustomerID")["InvoiceNo"].nunique().rename("Frequency")
        if "InvoiceNo" in df.columns else df.groupby("CustomerID")["InvoiceDate"].nunique().rename("Frequency")
    )
    rfm["Frequency"] = rfm["CustomerID"].map(order_frequency)
    rfm = rfm[["CustomerID", "Recency", "Frequency", "Monetary"]]

    print("\nRFM features preview:")
    print(rfm.head())
    return rfm


def plot_distribution(data: pd.DataFrame) -> None:
    """Create simple distribution plots for the RFM features."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.histplot(data["Recency"], bins=20, kde=True, ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("Recency Distribution")

    sns.histplot(data["Frequency"], bins=20, kde=True, ax=axes[0, 1], color="darkgreen")
    axes[0, 1].set_title("Frequency Distribution")

    sns.histplot(data["Monetary"], bins=20, kde=True, ax=axes[1, 0], color="darkorange")
    axes[1, 0].set_title("Monetary Distribution")

    sns.boxplot(data=data[["Recency", "Frequency", "Monetary"]], orient="h", ax=axes[1, 1])
    axes[1, 1].set_title("RFM Boxplots")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rfm_eda.png", dpi=300)
    plt.close()

    corr = data[["Recency", "Frequency", "Monetary"]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("RFM Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rfm_correlation_heatmap.png", dpi=300)
    plt.close()


def handle_outliers(rfm: pd.DataFrame) -> pd.DataFrame:
    """Clip extreme values using the IQR method."""
    cleaned = rfm.copy()

    for feature in ["Recency", "Frequency", "Monetary"]:
        q1 = cleaned[feature].quantile(0.25)
        q3 = cleaned[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        cleaned[feature] = cleaned[feature].clip(lower=lower_bound, upper=upper_bound)

    print("\nSummary after outlier handling:")
    print(cleaned.describe().round(2))
    return cleaned


def compute_kmeans_analysis(features: pd.DataFrame) -> tuple[int, pd.DataFrame, list[float], list[float]]:
    """Evaluate multiple K values using inertia and silhouette score."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    ks = range(2, 11)
    inertias = []
    silhouette_scores = []

    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled_features)
        inertias.append(model.inertia_)
        silhouette_scores.append(silhouette_score(scaled_features, labels))

    elbow_df = pd.DataFrame({"K": list(ks), "Inertia": inertias, "SilhouetteScore": silhouette_scores})

    plt.figure(figsize=(10, 5))
    plt.plot(elbow_df["K"], elbow_df["Inertia"], marker="o")
    plt.title("Elbow Method for K-Means")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.xticks(list(ks))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "elbow_method.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(elbow_df["K"], elbow_df["SilhouetteScore"], marker="o", color="purple")
    plt.title("Silhouette Score by K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.xticks(list(ks))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "silhouette_scores.png", dpi=300)
    plt.close()

    chosen_k = int(elbow_df.loc[elbow_df["SilhouetteScore"].idxmax(), "K"])
    print("\nK values evaluation:")
    print(elbow_df.round(4))
    print(f"\nSelected K: {chosen_k}")

    return chosen_k, elbow_df, inertias, silhouette_scores


def assign_segments(cluster_summary: pd.DataFrame) -> dict[int, str]:
    """Give each cluster a meaningful business-friendly label."""
    cluster_means = cluster_summary.copy()
    segment_map = {}

    for cluster_id, row in cluster_means.iterrows():
        recency = row["Recency"]
        frequency = row["Frequency"]
        monetary = row["Monetary"]

        if monetary >= cluster_means["Monetary"].median() and frequency >= cluster_means["Frequency"].median():
            segment_map[int(cluster_id)] = "High-Value Customers"
        elif recency <= cluster_means["Recency"].median() and frequency >= cluster_means["Frequency"].median():
            segment_map[int(cluster_id)] = "Regular Customers"
        elif recency <= cluster_means["Recency"].median() and monetary < cluster_means["Monetary"].median():
            segment_map[int(cluster_id)] = "New/Promising Customers"
        else:
            segment_map[int(cluster_id)] = "At-Risk Customers"

    return segment_map


def visualize_clusters(df: pd.DataFrame) -> None:
    """Create final cluster visualizations.

    PCA is used only to project the scaled RFM features into 2D for easier visualization.
    """
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[["Recency", "Frequency", "Monetary"]])

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_values)
    pca_df = pd.DataFrame(principal_components, columns=["PC1", "PC2"])
    pca_df["Cluster"] = df["Cluster"].astype(int)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="Set2", s=100)
    plt.title("Customer Clusters (PCA Projection)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "customer_cluster_scatter.png", dpi=300)
    plt.close()

    cluster_sizes = df["Cluster"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, palette="viridis")
    plt.title("Cluster Size Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_size_bar_chart.png", dpi=300)
    plt.close()

    summary = df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().reset_index()
    summary_melt = summary.melt(id_vars="Cluster", var_name="Feature", value_name="Value")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary_melt, x="Cluster", y="Value", hue="Feature", palette="deep")
    plt.title("Average RFM Comparison Across Clusters")
    plt.ylabel("Average Value")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rfm_cluster_comparison.png", dpi=300)
    plt.close()


def main() -> None:
    """Run the complete project workflow."""
    transaction_data = load_and_clean_data(DATA_PATH)
    rfm = build_rfm_features(transaction_data)

    plot_distribution(rfm)
    rfm = handle_outliers(rfm)

    feature_columns = ["Recency", "Frequency", "Monetary"]
    selected_features = rfm[feature_columns].copy()

    k_value, _, _, _ = compute_kmeans_analysis(selected_features)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_features)

    model = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_data)
    rfm["Cluster"] = labels + 1

    cluster_summary = (
        rfm.groupby("Cluster", as_index=False)[["Recency", "Frequency", "Monetary"]]
        .mean()
        .sort_values(["Monetary", "Frequency"], ascending=[False, False])
        .round(2)
    )

    segment_lookup = assign_segments(cluster_summary.set_index("Cluster"))
    rfm["Segment"] = rfm["Cluster"].map(segment_lookup)

    print("\nCluster summary with average RFM values:")
    print(cluster_summary)
    print("\nSegment mapping:")
    print(segment_lookup)

    final_output = rfm[["CustomerID", "Recency", "Frequency", "Monetary", "Cluster", "Segment"]].copy()
    final_output = final_output.sort_values("CustomerID").reset_index(drop=True)

    final_output.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)
    cluster_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)

    visualize_clusters(final_output)

    print(f"\nFinal segmented customer data saved to: {OUTPUT_DIR / 'customer_segments.csv'}")
    print(f"Cluster summary saved to: {OUTPUT_DIR / 'cluster_summary.csv'}")


if __name__ == "__main__":
    main()
