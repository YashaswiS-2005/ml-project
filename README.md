# E-Commerce Customer Segmentation

An interactive Streamlit dashboard that uses K-Means to group customers by Recency, Frequency, and Monetary value (RFM) from `ecommerce_user_segmentation.csv`. The supplied dataset contains 10,000 customer records. Other dataset columns, including `Segment_Label`, are not used to fit the model.

## Dashboard features

- Customer counts, selected cluster count, silhouette-recommended K, and champion count.
- RFM inputs to assign a new customer to the nearest fitted cluster.
- A PCA scatter plot, segment-size chart, and RFM cluster profile table.
- Marketing suggestions, a silhouette-score comparison for K values 2–6, and CSV export.
- An optional preview of the source data.

The model standardizes the three RFM features before fitting K-Means. Silhouette evaluation uses a reproducible sample of up to 2,000 customers for faster response; the clustering itself uses all valid customers.

Segment names such as “Champions” are heuristic labels assigned from relative RFM averages. They are not ground-truth categories or predictions of future behavior; review each segment profile before using the suggested actions. For new-customer prediction, zero Frequency and zero Monetary are treated as “No purchase history” rather than assigning a K-Means segment; Recency is not meaningful until a purchase occurs.

## Requirements and data validation

The CSV must include numeric, finite, non-negative `Recency`, `Frequency`, and `Monetary` columns, with at least six distinct RFM rows. Duplicate records are removed before clustering. The app displays an actionable error if the dataset is missing or invalid.

## Install and run (PowerShell)

Run these commands from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Run tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Project files

```text
app.py
segmentation_utils.py
tests/
requirements.txt
README.md
ecommerce_user_segmentation.csv
```
