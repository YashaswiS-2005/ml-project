# E-Commerce Customer Behaviour Clustering

A simple, beginner-friendly machine learning project that groups e-commerce customers by their spending, purchasing, and website behaviour.

## What it does

The Streamlit dashboard loads and cleans `dataset.csv`, selects numerical behaviour features, standardizes them with `StandardScaler`, and applies K-Means clustering. It includes an Elbow Method chart, customer-cluster visualization, cluster summary, and labelled customer table.

## Dataset columns

`Customer ID`, `Age`, `Gender`, `Annual Income`, `Number of Purchases`, `Total Spending`, `Average Order Value`, `Website/App Visits`, and `Time Spent on Website`.

The included CSV is small, fictional, and intended for learning.

## Run the project

Open a terminal in this folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Use the sidebar slider to try different values of K. The Elbow Method chart helps choose a suitable value; the default is 4.

## Customer segments

- High-Value Customers: highest spending and purchase activity.
- Regular Customers: consistent customers with solid activity.
- Occasional Customers: lower-frequency customers with some engagement.
- Low-Value Customers: lower spending and purchase activity.

The app ranks clusters by average spending and purchases before assigning these names.

## Project structure

```text
ecommerce-customer-clustering/
├── app.py
├── dataset.csv
├── requirements.txt
└── README.md
```

## Libraries

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, and Streamlit.
