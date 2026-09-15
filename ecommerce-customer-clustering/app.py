"""Streamlit dashboard for E-Commerce Customer Behaviour Clustering.

Run with: streamlit run app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Behaviour Clustering", layout="wide")
DATA_PATH = Path(__file__).with_name("dataset.csv")
FEATURES = ["Annual Income", "Number of Purchases", "Total Spending", "Average Order Value", "Website/App Visits", "Time Spent on Website"]


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the CSV, remove duplicates, and clean numerical values."""
    data = pd.read_csv(DATA_PATH).drop_duplicates().copy()
    for column in FEATURES + ["Age"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=FEATURES).reset_index(drop=True)


def name_segments(summary: pd.DataFrame) -> dict[int, str]:
    """Give each cluster a simple name based on spending and purchases."""
    ranked = summary.sort_values(["Total Spending", "Number of Purchases"], ascending=False).index.tolist()
    names = ["High-Value Customers", "Regular Customers", "Occasional Customers", "Low-Value Customers"]
    return {cluster: names[min(position, len(names) - 1)] for position, cluster in enumerate(ranked)}


@st.cache_data
def build_clusters(n_clusters: int):
    """Scale features, run K-Means, and calculate Elbow Method values."""
    data = load_data()
    # Scaling prevents high-value columns, such as spending, from dominating K-Means.
    scaled_features = StandardScaler().fit_transform(data[FEATURES])
    elbow_k = list(range(2, 9))
    inertia = np.array(
        [KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled_features).inertia_ for k in elbow_k]
    ).tolist()

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data["Cluster"] = model.fit_predict(scaled_features)
    summary = data.groupby("Cluster")[FEATURES].mean().round(2)
    segment_names = name_segments(summary)
    data["Customer Segment"] = data["Cluster"].map(segment_names)
    summary.insert(0, "Customer Segment", summary.index.map(segment_names))
    summary.insert(1, "Customers", data["Cluster"].value_counts().reindex(summary.index).values)

    # PCA is only used to make a simple two-dimensional cluster visualization.
    points = PCA(n_components=2, random_state=42).fit_transform(scaled_features)
    data["PCA 1"], data["PCA 2"] = points[:, 0], points[:, 1]
    return data, summary, elbow_k, inertia


st.title("E-Commerce Customer Behaviour Clustering")
st.write("Group customers with K-Means using shopping activity, spending, and website behaviour.")
with st.sidebar:
    st.header("Clustering settings")
    cluster_count = st.slider("Number of clusters", 2, 6, 4)

customers, cluster_summary, elbow_k, inertia = build_clusters(cluster_count)
left, right = st.columns(2)
left.metric("Number of customers", len(customers))
right.metric("Selected clusters", cluster_count)

st.subheader("Dataset preview")
st.dataframe(load_data().head(10), use_container_width=True)

st.subheader("Elbow Method")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(elbow_k, inertia, marker="o", color="#2563eb")
ax.set(xlabel="Number of clusters (K)", ylabel="Inertia", xticks=elbow_k)
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)
st.caption("Look for where the curve starts flattening. The sidebar selects the K value used in the results.")

st.subheader("Customer cluster visualization")
fig, ax = plt.subplots(figsize=(9, 5))
sns.scatterplot(data=customers, x="PCA 1", y="PCA 2", hue="Customer Segment", palette="Set2", s=85, ax=ax)
ax.set_title("Customers shown in two dimensions (PCA)")
st.pyplot(fig)
plt.close(fig)

st.subheader("Cluster summary")
st.dataframe(cluster_summary.reset_index(names="Cluster"), use_container_width=True)
st.subheader("Customers with cluster labels")
st.dataframe(customers.drop(columns=["PCA 1", "PCA 2"]).sort_values("Customer ID"), use_container_width=True)

st.subheader("What the customer segments mean")
st.markdown("""
- **High-Value Customers:** strongest spending and purchase activity.
- **Regular Customers:** consistent customers with solid engagement.
- **Occasional Customers:** customers who shop less frequently but still show activity.
- **Low-Value Customers:** lower spending and purchase activity; they may need targeted offers.

The app ranks clusters by average spending and purchases before assigning these names.
""")
