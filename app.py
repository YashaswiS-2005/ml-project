"""A simple Streamlit dashboard for e-commerce customer clustering."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from segmentation_utils import (
    DEFAULT_FEATURES,
    MAX_CLUSTERS,
    has_no_purchase_history,
    load_customer_data,
    predict_customer_cluster,
    segment_names,
)


st.set_page_config(page_title="Customer Clustering", page_icon="🛍️", layout="wide")

DATA_PATH = Path(__file__).with_name("ecommerce_user_segmentation.csv")
MIN_CLUSTERS = 2
SILHOUETTE_SAMPLE_SIZE = 2000

st.markdown(
    """<style>
    .block-container {max-width: 1180px; padding-top: 2.6rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {background: #f7f9fc; border: 1px solid #e8edf5; border-radius: 14px; padding: 16px;}
    [data-testid="stMetric"] * {color: #14213d !important;}
    [data-testid="stMetricLabel"] p {color: #4b5563 !important;}
    h1 {color: #14213d;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return load_customer_data(DATA_PATH)


@st.cache_data
def evaluate_clusters(scaled: np.ndarray) -> pd.DataFrame:
    """Evaluate every selectable K; sample silhouette points for predictable runtime."""
    scores = []
    for candidate in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
        model = KMeans(n_clusters=candidate, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
        scores.append(
            {
                "K": candidate,
                "Inertia": model.inertia_,
                "Silhouette": silhouette_score(
                    scaled,
                    labels,
                    sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(scaled)),
                    random_state=42,
                ),
            }
        )
    return pd.DataFrame(scores)


@st.cache_data
def create_clusters(k: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = load_data()
    scaled = StandardScaler().fit_transform(data[list(DEFAULT_FEATURES)])
    evaluation = evaluate_clusters(scaled)

    result = data.copy()
    result["Cluster"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    profile = result.groupby("Cluster")[list(DEFAULT_FEATURES)].mean().round(1)
    profile.insert(0, "Customers", result["Cluster"].value_counts().reindex(profile.index))
    labels = segment_names(profile)
    result["Segment"] = result["Cluster"].map(labels)
    profile.insert(0, "Segment", profile.index.map(labels))

    points = PCA(n_components=2, random_state=42).fit_transform(scaled)
    result["PCA 1"], result["PCA 2"] = points[:, 0], points[:, 1]
    return result, profile, evaluation


@st.cache_resource
def fitted_model(k: int) -> tuple[StandardScaler, KMeans]:
    """Fit the same RFM model used by the dashboard for a new-customer assignment."""
    rfm = load_data()[list(DEFAULT_FEATURES)]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm)
    model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled)
    return scaler, model


try:
    data = load_data()
except (OSError, ValueError) as error:
    st.error(f"Could not load the customer dataset: {error}")
    st.stop()

with st.sidebar:
    st.header("Settings")
    k = st.slider("Number of clusters", MIN_CLUSTERS, MAX_CLUSTERS, 4)
    st.caption("The model uses Recency, Frequency, and Monetary value (RFM).")

customers, profile, evaluation = create_clusters(k)
recommended_k = int(evaluation.loc[evaluation["Silhouette"].idxmax(), "K"])
champion_count = int((customers["Segment"] == "Champions").sum())

st.title("Customer Clustering Dashboard")
st.write("Discover meaningful e-commerce customer groups using **K-Means clustering**.")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Customers analysed", f"{len(customers):,}")
metric_2.metric("Clusters", k)
metric_3.metric("Recommended K", recommended_k)
metric_4.metric("Champions identified", f"{champion_count:,}")

st.divider()
st.subheader("About the model")
st.markdown("**RFM customer behaviour** &nbsp;→&nbsp; **Scaling** &nbsp;→&nbsp; **K-Means clustering** &nbsp;→&nbsp; **Actionable customer segments**")
st.caption("We group customers by how recently, how often, and how much they purchase. Scaling gives each RFM feature equal importance before K-Means finds similar customer groups.")

actions = {
    "Champions": "Reward with VIP benefits, early access, and referral offers.",
    "Loyal Customers": "Use bundles and cross-sell campaigns to increase order value.",
    "Potential Customers": "Encourage a repeat purchase with personalized recommendations.",
    "At-Risk Customers": "Run a time-limited win-back campaign.",
    "Low-Engagement Customers": "Use low-cost reactivation emails and improve onboarding.",
    "Other Customers": "Review their behaviour before targeting with a tailored campaign.",
    "No purchase history": "Use a welcome or first-purchase offer; do not treat this customer as at risk of churn.",
}

with st.container(border=True):
    st.subheader("✨ Predict a new customer's segment")
    st.write("Enter RFM values to assign a new customer to their nearest cluster.")
    with st.form("customer_prediction"):
        input_1, input_2, input_3, input_4 = st.columns([1, 1, 1, .7])
        with input_1:
            new_recency = st.number_input("Recency (days)", min_value=0.0, value=float(data["Recency"].median()), step=1.0)
        with input_2:
            new_frequency = st.number_input("Frequency (orders)", min_value=0.0, value=float(data["Frequency"].median()), step=1.0)
        with input_3:
            new_monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=float(data["Monetary"].median()), step=100.0)
        with input_4:
            st.write("")
            st.write("")
            predict_clicked = st.form_submit_button("Predict")

if predict_clicked:
    if has_no_purchase_history(new_frequency, new_monetary):
        st.info("Customer status: **No purchase history**")
        st.write(
            "Frequency and Monetary are both zero, so there is no recorded purchase to segment. "
            "Recency is not meaningful until a purchase has occurred."
        )
        st.write(actions["No purchase history"])
    else:
        scaler, model = fitted_model(k)
        cluster = predict_customer_cluster(scaler, model, [new_recency, new_frequency, new_monetary])
        predicted_segment = profile.loc[cluster, "Segment"]
        st.success(f"Predicted segment: **{predicted_segment}** (Cluster {cluster})")
        st.write("This customer was assigned here because their standardized RFM values are closest to this cluster's learned customer profile.")
        comparison = pd.DataFrame(
            {
                "RFM measure": ["Recency (days)", "Frequency (orders)", "Monetary (total spend)"],
                "New customer": [new_recency, new_frequency, new_monetary],
                f"Average {predicted_segment}": [profile.loc[cluster, "Recency"], profile.loc[cluster, "Frequency"], profile.loc[cluster, "Monetary"]],
            }
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        recency_message = "recently" if new_recency <= profile.loc[cluster, "Recency"] else "less recently"
        st.caption(f"Compared with the average {predicted_segment} customer, this customer purchased {recency_message}; K-Means considered Recency, Frequency, and Monetary together before choosing the nearest cluster.")
        st.write(actions[predicted_segment])

st.divider()
left, right = st.columns([1.35, 1])
with left:
    st.subheader("Customer groups")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=customers, x="PCA 1", y="PCA 2", hue="Segment", palette="Set2", alpha=0.72, s=26, linewidth=0, ax=ax)
    ax.set(xlabel="Customer behaviour dimension 1", ylabel="Customer behaviour dimension 2")
    ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.despine()
    st.pyplot(fig, clear_figure=True)
    st.caption("Each dot is a customer. Nearby dots have similar purchasing behaviour.")

with right:
    st.subheader("Segment sizes")
    size_data = customers["Segment"].value_counts().reindex(profile["Segment"]).reset_index()
    size_data.columns = ["Segment", "Customers"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=size_data, y="Segment", x="Customers", hue="Segment", palette="Set2", legend=False, ax=ax)
    ax.bar_label(ax.containers[0], padding=3, fmt="%.0f")
    ax.set(xlabel="Number of customers", ylabel="")
    sns.despine()
    st.pyplot(fig, clear_figure=True)

st.subheader("What the clusters mean")
st.caption(
    "Segment names are heuristic labels based on relative RFM cluster averages, not categories learned "
    "from confirmed customer outcomes. Review each segment's metrics before acting on the suggestions."
)
display_profile = profile.reset_index()[["Cluster", "Segment", "Customers", "Recency", "Frequency", "Monetary"]]
display_profile.columns = ["Cluster", "Segment", "Customers", "Days since last order", "Orders", "Total spend"]
st.dataframe(display_profile, use_container_width=True, hide_index=True)

with st.expander("Recommended marketing actions"):
    for name in profile["Segment"]:
        st.write(f"**{name}:** {actions[name]}")

with st.expander("Why did we select this K?"):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(evaluation["K"], evaluation["Silhouette"], marker="o", color="#2563eb")
    ax.axvline(recommended_k, color="#16a34a", linestyle="--", label=f"Recommended K = {recommended_k}")
    ax.set(xlabel="Number of clusters", ylabel="Silhouette score", xticks=evaluation["K"])
    ax.legend()
    ax.grid(alpha=.2)
    st.pyplot(fig, clear_figure=True)
    st.caption(
        "A higher silhouette score indicates clearer separation. Scores use up to 2,000 sampled customers "
        "for faster evaluation; all customers are used when the dataset is smaller."
    )

export_columns = [column for column in customers if column not in ["PCA 1", "PCA 2"]]
st.download_button("Download cluster results (CSV)", customers[export_columns].to_csv(index=False).encode("utf-8"), "customer_cluster_results.csv", "text/csv")

with st.expander("View source dataset (optional)"):
    st.dataframe(data.head(100), use_container_width=True, hide_index=True)
