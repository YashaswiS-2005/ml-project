# E-Commerce Customer Segmentation

An interactive Streamlit K-Means project based on the customer-segmentation workflow in the referenced tutorial. It uses the supplied `ecommerce_user_segmentation.csv` dataset to create actionable customer groups from RFM and web-engagement behaviour.

## Dataset support

The project uses `ecommerce_user_segmentation.csv`, containing 10,000 customer records and RFM, purchase, session, campaign, and cart-abandonment signals.

## What the dashboard does

1. Provides a dataset card with preview, data types, and missing values.
2. Explores feature distributions and correlations.
3. Lets you choose clustering features and K interactively.
4. Standardizes features with `StandardScaler` and applies K-Means.
5. Compares K values with the Elbow Method and silhouette score.
6. Shows a PCA cluster plot, segment profiles, campaign suggestions, and a CSV export.

## Install and run

Run these exact PowerShell commands from this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Project files

```text
app.py
requirements.txt
README.md
ecommerce_user_segmentation.csv
```
