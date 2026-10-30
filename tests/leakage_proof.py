"""Leakage proof: drop all orders >= 2025-09-01, rebuild features, assert match."""
import sys
import pandas as pd
import numpy as np

# Load clean data
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
products = pd.read_csv("data/clean/products.csv")

CUTOFF = pd.Timestamp("2025-09-01")

# ---- Method A: what part_b.py does (filter completed before cutoff) ----
completed_before = orders[(orders['order_ts'] < CUTOFF) & (orders['status'] == 'completed')]
cohort_customers = completed_before['customer_id'].unique()

# Build features exactly as part_b.py does
last_order = completed_before.groupby('customer_id')['order_ts'].max().reset_index()
last_order['recency'] = (CUTOFF - last_order['order_ts']).dt.days
last_order = last_order[['customer_id', 'recency']]

freq = completed_before.groupby('customer_id')['order_id'].nunique().reset_index()
freq.columns = ['customer_id', 'frequency']

mon = completed_before.groupby('customer_id')['order_amount'].sum().reset_index()
mon.columns = ['customer_id', 'monetary']

avg_disc = completed_before.groupby('customer_id')['discount_pct'].mean().reset_index()
avg_disc.columns = ['customer_id', 'avg_discount']

cancel_before = orders[(orders['order_ts'] < CUTOFF)]
total_orders = cancel_before.groupby('customer_id')['order_id'].count()
cancelled_orders = cancel_before[cancel_before['status'] == 'cancelled'].groupby('customer_id')['order_id'].count()
cancel_rate = (cancelled_orders / total_orders).fillna(0).reset_index()
cancel_rate.columns = ['customer_id', 'cancel_rate']

avg_del = completed_before.groupby('customer_id')['delivery_minutes'].mean().reset_index()
avg_del.columns = ['customer_id', 'avg_delivery_minutes']

orders_with_cat = completed_before.merge(products[['product_id', 'category']], on='product_id', how='left')

def most_frequent_alpha(series):
    counts = series.value_counts()
    max_count = counts.max()
    tied = counts[counts == max_count].index.tolist()
    return sorted(tied)[0]

fav_cat = orders_with_cat.groupby('customer_id')['category'].agg(most_frequent_alpha).reset_index()
fav_cat.columns = ['customer_id', 'fav_category']

cohort = pd.DataFrame({'customer_id': cohort_customers})
cohort = cohort.merge(last_order, on='customer_id')
cohort = cohort.merge(freq, on='customer_id')
cohort = cohort.merge(mon, on='customer_id')
cohort = cohort.merge(avg_disc, on='customer_id')
cohort = cohort.merge(cancel_rate, on='customer_id', how='left')
cohort['cancel_rate'] = cohort['cancel_rate'].fillna(0)
cohort = cohort.merge(avg_del, on='customer_id')
cohort = cohort.merge(fav_cat, on='customer_id', how='left')
cohort = cohort.merge(customers[['customer_id', 'city', 'gender', 'age', 'signup_date']], on='customer_id')
cohort['tenure_days'] = (CUTOFF - cohort['signup_date']).dt.days

feature_cols = ['recency', 'frequency', 'monetary', 'avg_discount', 'cancel_rate',
                'avg_delivery_minutes', 'age', 'tenure_days']
cat_cols = ['fav_category', 'city', 'gender']
features_A = cohort.sort_values('customer_id')[feature_cols + cat_cols].reset_index(drop=True)

# ---- Method B: physically delete Sept+ orders then rebuild identically ----
orders_no_sept = orders[orders['order_ts'] < CUTOFF].copy()
completed_no_sept = orders_no_sept[orders_no_sept['status'] == 'completed']
cohort_b = completed_no_sept['customer_id'].unique()

last_b = completed_no_sept.groupby('customer_id')['order_ts'].max().reset_index()
last_b['recency'] = (CUTOFF - last_b['order_ts']).dt.days
last_b = last_b[['customer_id', 'recency']]

freq_b = completed_no_sept.groupby('customer_id')['order_id'].nunique().reset_index()
freq_b.columns = ['customer_id', 'frequency']

mon_b = completed_no_sept.groupby('customer_id')['order_amount'].sum().reset_index()
mon_b.columns = ['customer_id', 'monetary']

avg_disc_b = completed_no_sept.groupby('customer_id')['discount_pct'].mean().reset_index()
avg_disc_b.columns = ['customer_id', 'avg_discount']

total_b = orders_no_sept.groupby('customer_id')['order_id'].count()
cancel_b = orders_no_sept[orders_no_sept['status'] == 'cancelled'].groupby('customer_id')['order_id'].count()
cr_b = (cancel_b / total_b).fillna(0).reset_index()
cr_b.columns = ['customer_id', 'cancel_rate']

avg_del_b = completed_no_sept.groupby('customer_id')['delivery_minutes'].mean().reset_index()
avg_del_b.columns = ['customer_id', 'avg_delivery_minutes']

oc_b = completed_no_sept.merge(products[['product_id', 'category']], on='product_id', how='left')
fav_b = oc_b.groupby('customer_id')['category'].agg(most_frequent_alpha).reset_index()
fav_b.columns = ['customer_id', 'fav_category']

cohort_df_b = pd.DataFrame({'customer_id': cohort_b})
cohort_df_b = cohort_df_b.merge(last_b, on='customer_id')
cohort_df_b = cohort_df_b.merge(freq_b, on='customer_id')
cohort_df_b = cohort_df_b.merge(mon_b, on='customer_id')
cohort_df_b = cohort_df_b.merge(avg_disc_b, on='customer_id')
cohort_df_b = cohort_df_b.merge(cr_b, on='customer_id', how='left')
cohort_df_b['cancel_rate'] = cohort_df_b['cancel_rate'].fillna(0)
cohort_df_b = cohort_df_b.merge(avg_del_b, on='customer_id')
cohort_df_b = cohort_df_b.merge(fav_b, on='customer_id', how='left')
cohort_df_b = cohort_df_b.merge(customers[['customer_id', 'city', 'gender', 'age', 'signup_date']], on='customer_id')
cohort_df_b['tenure_days'] = (CUTOFF - cohort_df_b['signup_date']).dt.days

features_B = cohort_df_b.sort_values('customer_id')[feature_cols + cat_cols].reset_index(drop=True)

# ---- Assert equality ----
print(f"Method A shape: {features_A.shape}")
print(f"Method B shape: {features_B.shape}")
print(f"Shapes equal: {features_A.shape == features_B.shape}")

# Compare numeric columns
for col in feature_cols:
    match = np.allclose(features_A[col].values, features_B[col].values, equal_nan=True)
    print(f"  {col}: {'MATCH' if match else 'MISMATCH'}")

# Compare categorical columns
for col in cat_cols:
    match = (features_A[col] == features_B[col]).all()
    print(f"  {col}: {'MATCH' if match else 'MISMATCH'}")

all_match = True
for col in feature_cols:
    if not np.allclose(features_A[col].values, features_B[col].values, equal_nan=True):
        all_match = False
for col in cat_cols:
    if not (features_A[col] == features_B[col]).all():
        all_match = False

print()
if all_match:
    print("LEAKAGE PROOF PASSED: Filtering orders < 2025-09-01 produces")
    print("identical features whether Sept+ orders exist in the dataset or not.")
    print("No September or later data leaks into the feature matrix.")
    sys.exit(0)
else:
    print("LEAKAGE PROOF FAILED!")
    sys.exit(1)
