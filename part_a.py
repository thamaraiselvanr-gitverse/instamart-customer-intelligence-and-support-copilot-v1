"""Part A - Data Foundation: Clean, Store, Query
Audits raw data, cleans it, stores in SQLite, and runs SQL analysis queries."""

import pandas as pd
import numpy as np
import sqlite3
import os

# ============================================================
# TASK 2: AUDIT THE RAW DATA
# ============================================================
print("=" * 60)
print("PART A: DATA FOUNDATION")
print("=" * 60)

# Load raw data
customers_raw = pd.read_csv("data/raw/customers.csv")
products_raw = pd.read_csv("data/raw/products.csv")
orders_raw = pd.read_csv("data/raw/orders.csv", keep_default_na=False)

print("\n--- RAW DATA AUDIT ---")

# Customers audit
print(f"\nCustomers: shape={customers_raw.shape}, dtypes:")
print(customers_raw.dtypes)
print("Missing values:")
for col in customers_raw.columns:
    missing = customers_raw[col].isna().sum() + (customers_raw[col].astype(str).str.strip() == '').sum()
    pct = missing / len(customers_raw) * 100
    print(f"  {col}: {missing} ({pct:.1f}%)")

# Products audit
print(f"\nProducts: shape={products_raw.shape}, dtypes:")
print(products_raw.dtypes)
print("Missing values:")
for col in products_raw.columns:
    missing = products_raw[col].isna().sum()
    pct = missing / len(products_raw) * 100
    print(f"  {col}: {missing} ({pct:.1f}%)")

# Orders audit
print(f"\nOrders: shape={orders_raw.shape}, dtypes:")
print(orders_raw.dtypes)
print("Missing values:")
for col in orders_raw.columns:
    missing = (orders_raw[col].astype(str).str.strip() == '').sum()
    pct = missing / len(orders_raw) * 100
    print(f"  {col}: {missing} ({pct:.1f}%)")

# Specific audit figures
print("\n--- SPECIFIC AUDIT FIGURES ---")

# Orders
n_raw_rows = len(orders_raw)
print(f"Raw orders.csv rows: {n_raw_rows}")

# Exact duplicates
dup_mask = orders_raw.duplicated(keep='first')
n_exact_dups = dup_mask.sum()
print(f"Exact duplicate rows: {n_exact_dups}")

# Pending rows
n_pending = (orders_raw['status'] == 'pending').sum()
print(f"Pending rows: {n_pending}")

# Distinct order_id
n_distinct_oid = orders_raw['order_id'].nunique()
print(f"Distinct order_id values: {n_distinct_oid}")

# Non-numeric order_amount
def is_non_numeric(val):
    try:
        float(val)
        return False
    except (ValueError, TypeError):
        return True

non_numeric_mask = orders_raw['order_amount'].apply(is_non_numeric)
n_non_numeric = non_numeric_mask.sum()
print(f"Non-numeric order_amount: {n_non_numeric}")

# Negative delivery_minutes
def parse_delivery(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

delivery_parsed = orders_raw['delivery_minutes'].apply(parse_delivery)
n_negative_delivery = (delivery_parsed < 0).sum()
print(f"Negative delivery_minutes: {n_negative_delivery}")

n_blank_delivery = (orders_raw['delivery_minutes'].astype(str).str.strip() == '').sum()
print(f"Blank delivery_minutes: {n_blank_delivery}")

# Customers specifics
# Blank age: pandas reads blank numeric cells as NaN
n_blank_age = customers_raw['age'].isna().sum()
print(f"Blank age in customers: {n_blank_age}")

# Age outside 10-100
def parse_age(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

ages_parsed = customers_raw['age'].apply(parse_age)
valid_ages = ages_parsed.dropna()
n_age_outside = ((valid_ages < 10) | (valid_ages > 100)).sum()
print(f"Age values outside 10-100: {n_age_outside}")

# Distinct raw city spellings
n_city_spellings = customers_raw['city'].nunique()
print(f"Distinct raw city spellings: {n_city_spellings}")

# Distinct non-blank raw gender spellings
gender_filled = customers_raw['gender'].fillna('').astype(str).str.strip()
gender_nonblank = gender_filled[gender_filled != '']
n_gender_spellings = gender_nonblank.nunique()
print(f"Distinct non-blank raw gender spellings: {n_gender_spellings}")

# Blank gender
gender_str = customers_raw['gender'].fillna('').astype(str).str.strip()
n_blank_gender = (gender_str == '').sum()
print(f"Blank gender: {n_blank_gender}")


# ============================================================
# TASK 3: CLEAN CUSTOMERS
# ============================================================
print("\n--- CLEANING CUSTOMERS ---")
customers = customers_raw.copy()

# (a) Standardise city
customers['city'] = customers['city'].str.strip().str.title()
print(f"Distinct cities after cleaning: {customers['city'].nunique()}")
print(f"City counts:\n{customers['city'].value_counts().sort_index()}")

# (b) Map gender
gender_map = {
    'male': 'male', 'Male': 'male', 'M': 'male', 'm': 'male', 'MALE': 'male',
    'female': 'female', 'Female': 'female', 'F': 'female', 'f': 'female', 'FEMALE': 'female',
}
customers['gender'] = customers['gender'].astype(str).str.strip()
customers['gender'] = customers['gender'].map(lambda g: gender_map.get(g, 'unknown'))
print(f"Gender counts:\n{customers['gender'].value_counts().sort_index()}")

# (c) Clean age - treat outside 10-100 as missing, fill with city median
customers['age'] = pd.to_numeric(customers['age'], errors='coerce')
customers.loc[(customers['age'] < 10) | (customers['age'] > 100), 'age'] = np.nan
n_missing_age_before = customers['age'].isna().sum()
print(f"Missing ages before imputation: {n_missing_age_before}")

city_median_age = customers.groupby('city')['age'].transform('median')
customers['age'] = customers['age'].fillna(city_median_age)
# Convert to int
customers['age'] = customers['age'].astype(int)
print(f"Missing ages after imputation: {customers['age'].isna().sum()}")
print(f"Age range: {customers['age'].min()} - {customers['age'].max()}")

# (d) Convert signup_date
customers['signup_date'] = pd.to_datetime(customers['signup_date'])

print(f"\nCleaned customers: {len(customers)} rows")


# ============================================================
# TASK 4: CLEAN ORDERS
# ============================================================
print("\n--- CLEANING ORDERS ---")
orders = orders_raw.copy()

# Step 1: Convert order_ts to datetime
orders['order_ts'] = pd.to_datetime(orders['order_ts'])
print(f"Step 1 - Converted order_ts to datetime")

# Step 2: Remove exact duplicate rows
n_before = len(orders)
orders = orders.drop_duplicates()
n_after = len(orders)
print(f"Step 2 - Removed {n_before - n_after} exact duplicate rows: {n_before} -> {n_after}")

# Step 3: For each order_id, keep only the record with the latest order_ts
n_before = len(orders)
orders = orders.sort_values('order_ts').drop_duplicates(subset='order_id', keep='last')
n_after = len(orders)
print(f"Step 3 - Removed {n_before - n_after} duplicate order_ids (kept latest): {n_before} -> {n_after}")

# Step 4: Convert order_amount to numeric, fill missing from product price
orders['order_amount'] = pd.to_numeric(orders['order_amount'], errors='coerce')
n_missing_amount = orders['order_amount'].isna().sum()
print(f"Step 4 - Non-numeric order_amount converted to NaN: {n_missing_amount}")

# Merge with products to get price_inr
products = products_raw.copy()
orders = orders.merge(products[['product_id', 'price_inr']], on='product_id', how='left')
orders['discount_pct'] = pd.to_numeric(orders['discount_pct'], errors='coerce').fillna(0)
orders['quantity'] = pd.to_numeric(orders['quantity'], errors='coerce').fillna(1).astype(int)

# Fill missing amounts
fill_mask = orders['order_amount'].isna()
orders.loc[fill_mask, 'order_amount'] = (
    orders.loc[fill_mask, 'price_inr'] * orders.loc[fill_mask, 'quantity'] *
    (1 - orders.loc[fill_mask, 'discount_pct'] / 100)
).round(2)
print(f"Step 4 - Filled {fill_mask.sum()} missing amounts from product price")
orders = orders.drop(columns=['price_inr'])

# Step 5: Fix delivery_minutes
orders['delivery_minutes'] = pd.to_numeric(orders['delivery_minutes'], errors='coerce')

# Set negative to missing
n_negative = (orders['delivery_minutes'] < 0).sum()
orders.loc[orders['delivery_minutes'] < 0, 'delivery_minutes'] = np.nan
print(f"Step 5 - Set {n_negative} negative delivery_minutes to NaN")

# For completed orders, fill missing delivery_minutes with category median
# Need to merge product category
orders = orders.merge(products[['product_id', 'category']], on='product_id', how='left')

completed_mask = orders['status'] == 'completed'
missing_delivery_completed = completed_mask & orders['delivery_minutes'].isna()
n_missing_del_before = missing_delivery_completed.sum()

# Compute category median only from completed orders
cat_median = orders.loc[completed_mask & orders['delivery_minutes'].notna()].groupby('category')['delivery_minutes'].transform('median')
# We need to align properly - use groupby transform on the full orders df but only completed
orders['cat_median_delivery'] = orders.merge(
    orders.loc[completed_mask & orders['delivery_minutes'].notna()].groupby('category')['delivery_minutes'].median().reset_index().rename(columns={'delivery_minutes': 'cat_med'}),
    on='category', how='left'
)['cat_med']

orders.loc[missing_delivery_completed, 'delivery_minutes'] = orders.loc[missing_delivery_completed, 'cat_median_delivery']
orders = orders.drop(columns=['cat_median_delivery', 'category'])

n_missing_del_after = (completed_mask & orders['delivery_minutes'].isna()).sum()
print(f"Step 5 - Filled {n_missing_del_before - n_missing_del_after} missing delivery_minutes for completed orders")

# For cancelled orders, leave delivery_minutes as NaN
cancelled_mask = orders['status'] == 'cancelled'
print(f"Step 5 - Cancelled orders with missing delivery: {(cancelled_mask & orders['delivery_minutes'].isna()).sum()}")

# Final validation
print(f"\n--- CLEANED ORDERS SUMMARY ---")
print(f"Total rows: {len(orders)}")
print(f"Completed: {(orders['status'] == 'completed').sum()}")
print(f"Cancelled: {(orders['status'] == 'cancelled').sum()}")
print(f"Pending: {(orders['status'] == 'pending').sum()}")
print(f"Missing order_amount: {orders['order_amount'].isna().sum()}")
print(f"Missing delivery_minutes (completed): {(completed_mask & orders['delivery_minutes'].isna()).sum()}")
print(f"Missing delivery_minutes (cancelled): {(cancelled_mask & orders['delivery_minutes'].isna()).sum()}")
completed_sum = orders.loc[orders['status'] == 'completed', 'order_amount'].sum()
print(f"Sum of order_amount (completed): INR {completed_sum:.2f}")


# ============================================================
# TASK 5: STORE AND VALIDATE
# ============================================================
print("\n--- STORING CLEAN DATA ---")
os.makedirs("data/clean", exist_ok=True)

# Save clean CSVs
customers.to_csv("data/clean/customers.csv", index=False)
products.to_csv("data/clean/products.csv", index=False)

# Format order_ts as string for CSV
orders_save = orders.copy()
orders_save['order_ts'] = orders_save['order_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
orders_save.to_csv("data/clean/orders.csv", index=False)

# Save to SQLite
db_path = "data/instamart.db"
if os.path.exists(db_path):
    os.remove(db_path)
conn = sqlite3.connect(db_path)

# Customers - convert signup_date to string
cust_db = customers.copy()
cust_db['signup_date'] = cust_db['signup_date'].dt.strftime('%Y-%m-%d')
cust_db.to_sql('customers', conn, if_exists='replace', index=False)

products.to_sql('products', conn, if_exists='replace', index=False)

orders_db = orders.copy()
orders_db['order_ts'] = orders_db['order_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
orders_db.to_sql('orders', conn, if_exists='replace', index=False)

conn.close()
print(f"Saved to {db_path}")
print("Clean CSVs saved to data/clean/")


# ============================================================
# TASK 6: SQL ANALYSIS
# ============================================================
print("\n--- SQL ANALYSIS ---")
os.makedirs("sql", exist_ok=True)

conn = sqlite3.connect(db_path)

# Q1 - WHERE then HAVING
q1 = """
-- Q1: WHERE then HAVING
-- Keep only completed orders with order_amount > 500, group by customer,
-- keep customers whose total exceeds 8000.
SELECT customer_id, SUM(order_amount) AS total_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;
"""

# Q2 - Three-table join: Revenue by city and product category
q2 = """
-- Q2: Three-table JOIN
-- Revenue by city and product category (completed orders only).
SELECT c.city, p.category, SUM(o.order_amount) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;
"""

# Q3 - Scalar subquery: Products priced above average catalogue price
q3 = """
-- Q3: Scalar subquery
-- Products priced above the average catalogue price.
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;
"""

# Q4 - LEFT JOIN: Customers with no completed order
q4 = """
-- Q4: LEFT JOIN
-- Customers who have no completed order.
SELECT c.customer_id
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;
"""

# Q5 - CTE: Customers above average total spend
q5 = """
-- Q5: Common Table Expression
-- Customers whose total completed spend is above average.
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) AS avg_total
    FROM customer_spend
)
SELECT cs.customer_id, cs.total_spend
FROM customer_spend cs, avg_spend a
WHERE cs.total_spend > a.avg_total
ORDER BY cs.total_spend DESC;
"""

# Q6 - Ranking window functions
q6 = """
-- Q6: Ranking window functions
-- For each customer, compute total spend and RANK(), DENSE_RANK(), ROW_NUMBER()
-- partitioned by city, ordered by total_spend DESC.
SELECT customer_id, city, total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM (
    SELECT o.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY o.customer_id, c.city
) sub
ORDER BY city, total_spend DESC;
"""

# Q7 - Running total: Daily completed revenue with cumulative sum
q7 = """
-- Q7: Running total
-- Daily completed revenue with cumulative revenue.
SELECT day, revenue,
       SUM(revenue) OVER (ORDER BY day) AS cumulative_revenue
FROM (
    SELECT SUBSTR(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
) daily
ORDER BY day;
"""

# Save all queries to file
with open("sql/queries.sql", "w", encoding="utf-8") as f:
    f.write(q1 + "\n" + q2 + "\n" + q3 + "\n" + q4 + "\n" + q5 + "\n" + q6 + "\n" + q7)

# Execute and report
print("\nQ1 - Customers with total high-value spend > 8000:")
df_q1 = pd.read_sql_query(q1, conn)
print(f"  Number of customers: {len(df_q1)}")

print("\nQ2 - Revenue by city and category:")
df_q2 = pd.read_sql_query(q2, conn)
print(f"  Number of rows: {len(df_q2)}")
print(f"  Total revenue: INR {df_q2['revenue'].sum():.2f}")
print(f"  Largest cell: {df_q2.loc[df_q2['revenue'].idxmax(), 'city']} / {df_q2.loc[df_q2['revenue'].idxmax(), 'category']} at INR {df_q2['revenue'].max():.2f}")

print("\nQ3 - Products above average price:")
df_q3 = pd.read_sql_query(q3, conn)
print(f"  Number of products: {len(df_q3)}")

print("\nQ4 - Customers with no completed order:")
df_q4 = pd.read_sql_query(q4, conn)
print(f"  Number of customers: {len(df_q4)}")

print("\nQ5 - Customers above average spend (CTE):")
df_q5 = pd.read_sql_query(q5, conn)
print(f"  Number of customers: {len(df_q5)}")
# Compute average
avg_q5 = pd.read_sql_query("""
    WITH customer_spend AS (
        SELECT customer_id, SUM(order_amount) AS total_spend
        FROM orders WHERE status = 'completed' GROUP BY customer_id
    )
    SELECT AVG(total_spend) AS avg_total FROM customer_spend
""", conn)
print(f"  Average total spend: INR {avg_q5['avg_total'].iloc[0]:.2f}")

print("\nQ6 - Ranking window functions:")
df_q6 = pd.read_sql_query(q6, conn)
# Top customer per city
top_per_city = df_q6[df_q6['rnk'] == 1][['city', 'customer_id', 'total_spend']]
for _, row in top_per_city.iterrows():
    print(f"  {row['city']}: {row['customer_id']} ({row['total_spend']:.2f})")
# Count where RANK != ROW_NUMBER
rank_diff = (df_q6['rnk'] != df_q6['row_num']).sum()
print(f"  Rows where RANK() != ROW_NUMBER(): {rank_diff}")

print("\nQ7 - Running total (daily completed revenue):")
df_q7 = pd.read_sql_query(q7, conn)
print(f"  Number of daily rows: {len(df_q7)}")
print(f"  Last cumulative value: INR {df_q7['cumulative_revenue'].iloc[-1]:.2f}")

conn.close()


# ============================================================
# TASK 7: PANDAS CROSS-CHECK AND EXPLORATORY ANALYSIS
# ============================================================
print("\n--- PANDAS CROSS-CHECK ---")

# Load clean data
orders_clean = pd.read_csv("data/clean/orders.csv")
customers_clean = pd.read_csv("data/clean/customers.csv")
products_clean = pd.read_csv("data/clean/products.csv")

# (a) Rebuild Q2 with pivot_table
merged = orders_clean[orders_clean['status'] == 'completed'].merge(
    customers_clean[['customer_id', 'city']], on='customer_id'
).merge(
    products_clean[['product_id', 'category']], on='product_id'
)

pivot = pd.pivot_table(merged, values='order_amount', index='city', columns='category',
                       aggfunc='sum', margins=True)
print(f"\nPivot table grand total: INR {pivot.loc['All', 'All']:.2f}")

# Per-city stats with named aggregation
city_stats = merged.groupby('city')['order_amount'].agg(
    total_sum='sum',
    total_mean='mean',
    total_count='count'
)
print(f"\nPer-city stats:")
print(city_stats.round(2))

# (b) Exploratory analysis of completed order_amount
completed = orders_clean[orders_clean['status'] == 'completed']
amount = completed['order_amount']

mean_val = amount.mean()
median_val = amount.median()
skew_val = amount.skew()
q1_val = amount.quantile(0.25)
q3_val = amount.quantile(0.75)
iqr = q3_val - q1_val
lower_fence = q1_val - 1.5 * iqr
upper_fence = q3_val + 1.5 * iqr
outliers_above = (amount > upper_fence).sum()
outliers_below = (amount < lower_fence).sum()

print(f"\n--- COMPLETED ORDER AMOUNT STATISTICS ---")
print(f"Mean: {mean_val:.2f}")
print(f"Median: {median_val:.2f}")
print(f"Skewness: {skew_val:.2f}")
print(f"Q1: {q1_val:.2f}")
print(f"Q3: {q3_val:.2f}")
print(f"IQR: {iqr:.2f}")
print(f"Lower fence: {lower_fence:.2f}")
print(f"Upper fence: {upper_fence:.2f}")
print(f"Outliers above upper fence: {outliers_above}")
print(f"Outliers below lower fence: {outliers_below}")

# Correlation matrix
corr_cols = ['order_amount', 'discount_pct', 'quantity', 'delivery_minutes']
corr_df = completed[corr_cols].astype(float)
corr_matrix = corr_df.corr()
print(f"\nCorrelation matrix:")
print(corr_matrix.round(3))

# Find largest correlation pair (absolute value, excluding diagonal)
import itertools
max_corr_pair = None
max_corr_val = 0
for c1, c2 in itertools.combinations(corr_cols, 2):
    val = abs(corr_matrix.loc[c1, c2])
    if val > max_corr_val:
        max_corr_val = val
        max_corr_pair = (c1, c2)
print(f"\nLargest correlation (absolute): {max_corr_pair[0]} and {max_corr_pair[1]} at {corr_matrix.loc[max_corr_pair[0], max_corr_pair[1]]:.3f}")

print("\nPart A complete.")
