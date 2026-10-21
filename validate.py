"""validate.py -- Asserts at least eight reference checkpoints from Part A."""

import pandas as pd
import sqlite3

# Load clean data
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

# Checkpoint 1: 12,309 rows (one per order_id)
assert len(orders) == 12309, f"Expected 12309 orders, got {len(orders)}"

# Checkpoint 2: 11,110 completed and 1,199 cancelled
assert (orders['status'] == 'completed').sum() == 11110, f"Expected 11110 completed"
assert (orders['status'] == 'cancelled').sum() == 1199, f"Expected 1199 cancelled"

# Checkpoint 3: no pending rows
assert (orders['status'] == 'pending').sum() == 0, "Expected 0 pending"

# Checkpoint 4: 0 missing order_amount
assert orders['order_amount'].isna().sum() == 0, "Expected 0 missing order_amount"

# Checkpoint 5: 0 missing delivery_minutes among completed orders
completed = orders[orders['status'] == 'completed']
assert completed['delivery_minutes'].isna().sum() == 0, "Expected 0 missing delivery_minutes for completed"

# Checkpoint 6: 1,199 missing delivery_minutes among cancelled
cancelled = orders[orders['status'] == 'cancelled']
assert cancelled['delivery_minutes'].isna().sum() == 1199, "Expected 1199 missing delivery_minutes for cancelled"

# Checkpoint 7: sum of completed order_amount = INR 2,998,668.80
completed_sum = round(completed['order_amount'].sum(), 2)
assert completed_sum == 2998668.80, f"Expected sum 2998668.80, got {completed_sum}"

# Checkpoint 8: 6 distinct cities
assert customers['city'].nunique() == 6, "Expected 6 distinct cities"

# Checkpoint 9: gender counts
assert (customers['gender'] == 'female').sum() == 610, "Expected 610 female"
assert (customers['gender'] == 'male').sum() == 560, "Expected 560 male"
assert (customers['gender'] == 'unknown').sum() == 30, "Expected 30 unknown"

# Checkpoint 10: 0 missing ages, all ages 18-60
assert customers['age'].isna().sum() == 0, "Expected 0 missing ages"
assert customers['age'].min() >= 18, f"Min age {customers['age'].min()} < 18"
assert customers['age'].max() <= 60, f"Max age {customers['age'].max()} > 60"

# Checkpoint 11: SQLite queries
conn = sqlite3.connect("data/instamart.db")
q1_count = pd.read_sql_query("""
    SELECT COUNT(*) as cnt FROM (
        SELECT customer_id FROM orders
        WHERE status = 'completed' AND order_amount > 500
        GROUP BY customer_id HAVING SUM(order_amount) > 8000
    )
""", conn).iloc[0, 0]
assert q1_count == 30, f"Q1: Expected 30, got {q1_count}"

q7_last = pd.read_sql_query("""
    SELECT SUM(revenue) OVER (ORDER BY day) AS cumulative_revenue
    FROM (
        SELECT SUBSTR(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
        FROM orders WHERE status = 'completed'
        GROUP BY SUBSTR(order_ts, 1, 10)
    ) ORDER BY day DESC LIMIT 1
""", conn).iloc[0, 0]
assert round(q7_last, 2) == 2998668.80, f"Q7 last cum: Expected 2998668.80, got {q7_last}"

conn.close()

print("ALL CHECKPOINTS PASSED")
