# Olist E-Commerce Customer Segmentation

An unsupervised machine learning project that segments customers from the Olist Brazilian e-commerce platform into behaviorally distinct groups using RFM analysis and K-Means clustering. The goal is to give a business team actionable customer groups they can target differently rather than treating all 93,000 customers the same way.

---

## Results

**Final Model: K-Means Clustering (K=3, Silhouette Score = 0.4129)**

| Segment | Count | Avg Recency | Avg Spend | Recommended Action |
|---|---|---|---|---|
| Lapsed High-Value | 38,893 | 278 days | $138.70 | Win-back campaign. High spend history makes them worth pursuing. |
| Lost Cheap | 29,980 | 281 days | $37.80 | Low priority. Consider a low-cost reactivation email only. |
| Active Customers | 14,373 | 40 days | $92.80 | Nurture and upsell. They are engaged right now. |

K=3 was selected using two independent methods. The Elbow curve bent sharply between K=2 and K=3, and the Silhouette score peaked at 0.4129 at K=3 before dropping to around 0.33 for every higher value.

---

## Dataset

- **Source:** Kaggle - Brazilian E-Commerce Public Dataset by Olist
- **Tables used:** olist_customers_dataset, olist_orders_dataset, olist_order_items_dataset
- **Raw size:** 93,358 unique customers after joining and filtering delivered orders
- **Final size:** 83,246 customers after outlier removal

---

## Folder Structure

```
olist-customer-segmentation/
├── data/
│   ├── olist_customers_dataset.csv
│   ├── olist_orders_dataset.csv
│   └── olist_order_items_dataset.csv
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── iqr_bounds.json
│   └── cluster_labels.json
├── notebook/
│   └── e-commerce.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Project Walkthrough

### 1. Data Joining

Three separate CSV files were joined to build a single customer-level view. Orders were merged with items on `order_id`, then merged with customers on `customer_id`. Only delivered orders were kept, filtering out cancelled, unavailable, and in-progress orders to ensure RFM values reflect completed transactions only.

```
customers (99,441 rows) + orders (99,441 rows) + items (112,650 rows)
→ full joined table: 110,197 rows, 18 columns
→ after filtering delivered only: 110,197 rows retained
```

### 2. RFM Feature Engineering

The data was aggregated from item-level rows to one row per customer using three behavioral metrics:

- **Recency** — days since the customer's last order, measured against the day after the most recent purchase in the dataset. Lower = more recent = better customer.
- **Frequency** — number of distinct orders placed. Counted using `order_id` unique values to avoid inflating the count with multi-item orders.
- **Monetary** — total sum of item prices across all orders.

`customer_unique_id` was used as the grouping key, not `customer_id`, because Olist assigns a new `customer_id` per order while `customer_unique_id` identifies the actual person.

### 3. Dropping Frequency

After computing RFM, `describe()` showed that over 90% of customers had a frequency of exactly 1. After log transform and outlier removal, frequency had a standard deviation of 0.0, meaning every customer mapped to the same value. A feature with zero variance cannot separate any customers from each other, so it was dropped before clustering. The final model was built on Recency and Monetary only.

### 4. Outlier Removal

Outliers were removed using the IQR method on all three features before dropping frequency. Values beyond Q3 + 1.5×IQR were clipped out rather than capped, as extreme values in an unsupervised setting distort centroid positions.

```
Recency:   removed 26 rows
Frequency: removed 2,801 rows
Monetary:  removed 7,285 rows
Final size after removal: 83,246 customers
```

The IQR bounds were saved to `iqr_bounds.json` so the same clipping logic can be applied to new inputs at prediction time.

### 5. Log Transform

Even after outlier removal, Recency and Monetary remained right-skewed. K-Means uses Euclidean distance, so skewed distributions bias distance calculations toward outlier-direction values. `np.log1p` was applied to both features to compress the long tail and produce more balanced distributions.

### 6. Scaling

`StandardScaler` was applied to bring both features to mean=0 and std=1. Without this step, Recency measured in hundreds of days would dominate distance calculations over Monetary values, regardless of actual importance.

---

## Finding the Optimal K

Two methods were used together to select the number of clusters rather than relying on either alone.

**Elbow Method** — K-Means was run for K=2 through K=10 and inertia was recorded at each step. Inertia dropped by approximately 47,000 from K=2 to K=3, then flattened significantly. The bend in the curve was clear and unambiguous at K=3.

**Silhouette Score** — Scores were computed at each K using a 10,000-row sample for efficiency. K=3 produced a score of 0.4129, a clear spike above the 0.33-0.35 range seen at every higher K.

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 109,356 | 0.3459 |
| **3** | **62,698** | **0.4129** |
| 4 | 51,005 | 0.3387 |
| 5 | 41,778 | 0.3382 |
| 6 | 34,830 | 0.3491 |
| 7 | 29,986 | 0.3482 |

---

## Saved Artifacts

| File | Description |
|---|---|
| `models/kmeans_model.pkl` | Trained K-Means model (K=3) |
| `models/scaler.pkl` | Fitted StandardScaler for preprocessing new inputs |
| `models/iqr_bounds.json` | IQR clip bounds for Recency and Monetary |
| `models/cluster_labels.json` | Mapping from cluster IDs (0, 1, 2) to segment names |
| `notebook/e-commerce.ipynb` | Full notebook with code and outputs |

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

Install with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

---

## How to Run

1. Clone the repo
2. The dataset files are already included in the data/ folder
3. Make sure your notebook is pointing to the correct relative paths: `../data/olist_customers_dataset.csv`
4. Open and run `e-commerce.ipynb` from top to bottom

---

## Key Takeaways

- RFM aggregation is the right starting point for e-commerce customer segmentation. It compresses hundreds of thousands of rows into three numbers that actually describe customer behavior.
- Always check feature variance before clustering. Frequency looked like a valid feature until the data showed 90% of customers had identical values. Including it would have added noise, not signal.
- Inertia alone is a dishonest metric for choosing K. It always rewards more clusters. Silhouette score penalizes clusters that are not well separated, making it the more reliable of the two.
- The most commercially interesting segment here is Lapsed High-Value. Nearly 39,000 customers who once spent an average of $138 have gone quiet for almost a year. A targeted win-back campaign on this group has the clearest ROI of any action the business could take.
