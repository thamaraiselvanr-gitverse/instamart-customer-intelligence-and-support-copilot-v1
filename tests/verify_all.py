"""tests/verify_all.py -- Cumulative verification of all brief reference checkpoints.
Run after the four Part scripts to confirm every number matches.
Prints PASS/FAIL per check and exits with code 1 on any failure."""

import pandas as pd
import numpy as np
import sqlite3
import os
import json
import re
import sys

results = []

def check(name, condition, expected, actual):
    status = "PASS" if condition else "FAIL"
    results.append({"check": name, "expected": str(expected), "actual": str(actual), "status": status})
    if not condition:
        print(f"  FAIL: {name} -- expected {expected}, got {actual}")

# ============================================================
# PART A CHECKPOINTS
# ============================================================
print("--- Part A Checks ---")

# Load clean data
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")
orders['order_ts'] = pd.to_datetime(orders['order_ts'])
customers['signup_date'] = pd.to_datetime(customers['signup_date'])

# A1: 12,309 rows
check("A1: clean orders row count", len(orders) == 12309, 12309, len(orders))

# A2: 11,110 completed, 1,199 cancelled
n_comp = (orders['status'] == 'completed').sum()
n_canc = (orders['status'] == 'cancelled').sum()
check("A2a: completed orders", n_comp == 11110, 11110, n_comp)
check("A2b: cancelled orders", n_canc == 1199, 1199, n_canc)

# A3: no pending
n_pend = (orders['status'] == 'pending').sum()
check("A3: no pending orders", n_pend == 0, 0, n_pend)

# A4: 0 missing order_amount
check("A4: no missing order_amount", orders['order_amount'].isna().sum() == 0, 0, orders['order_amount'].isna().sum())

# A5: 0 missing delivery_minutes in completed
completed = orders[orders['status'] == 'completed']
check("A5: no missing delivery (completed)", completed['delivery_minutes'].isna().sum() == 0, 0, completed['delivery_minutes'].isna().sum())

# A6: 1,199 missing delivery in cancelled
cancelled = orders[orders['status'] == 'cancelled']
check("A6: missing delivery (cancelled)", cancelled['delivery_minutes'].isna().sum() == 1199, 1199, cancelled['delivery_minutes'].isna().sum())

# A7: completed sum = 2,998,668.80
comp_sum = round(completed['order_amount'].sum(), 2)
check("A7: completed amount sum", comp_sum == 2998668.80, 2998668.80, comp_sum)

# A8: 6 cities
check("A8: 6 distinct cities", customers['city'].nunique() == 6, 6, customers['city'].nunique())

# A9: gender counts
check("A9a: female count", (customers['gender'] == 'female').sum() == 610, 610, (customers['gender'] == 'female').sum())
check("A9b: male count", (customers['gender'] == 'male').sum() == 560, 560, (customers['gender'] == 'male').sum())
check("A9c: unknown count", (customers['gender'] == 'unknown').sum() == 30, 30, (customers['gender'] == 'unknown').sum())

# A10: 0 missing ages, all 18-60
check("A10a: no missing age", customers['age'].isna().sum() == 0, 0, customers['age'].isna().sum())
check("A10b: min age >= 18", customers['age'].min() >= 18, True, customers['age'].min() >= 18)
check("A10c: max age <= 60", customers['age'].max() <= 60, True, customers['age'].max() <= 60)

# A11: city counts
city_counts = customers['city'].value_counts()
check("A11a: Pune count", city_counts.get('Pune', 0) == 227, 227, city_counts.get('Pune', 0))
check("A11b: Delhi count", city_counts.get('Delhi', 0) == 210, 210, city_counts.get('Delhi', 0))
check("A11c: Bengaluru count", city_counts.get('Bengaluru', 0) == 197, 197, city_counts.get('Bengaluru', 0))
check("A11d: Mumbai count", city_counts.get('Mumbai', 0) == 184, 184, city_counts.get('Mumbai', 0))

# SQL checks
conn = sqlite3.connect("data/instamart.db")

q1 = pd.read_sql_query("""
    SELECT customer_id FROM orders
    WHERE status = 'completed' AND order_amount > 500
    GROUP BY customer_id HAVING SUM(order_amount) > 8000
""", conn)
check("A-SQL-Q1: 30 customers", len(q1) == 30, 30, len(q1))

q2 = pd.read_sql_query("""
    SELECT c.city, p.category, SUM(o.order_amount) AS revenue
    FROM orders o JOIN customers c ON o.customer_id = c.customer_id
    JOIN products p ON o.product_id = p.product_id
    WHERE o.status = 'completed' GROUP BY c.city, p.category
""", conn)
check("A-SQL-Q2: 30 rows", len(q2) == 30, 30, len(q2))
blr_elec = q2[(q2['city'] == 'Bengaluru') & (q2['category'] == 'Electronics')]['revenue'].iloc[0]
check("A-SQL-Q2: Bengaluru Electronics", round(blr_elec, 2) == 233909.45, 233909.45, round(blr_elec, 2))

q3 = pd.read_sql_query("SELECT * FROM products WHERE price_inr > (SELECT AVG(price_inr) FROM products)", conn)
check("A-SQL-Q3: 11 products", len(q3) == 11, 11, len(q3))

q4 = pd.read_sql_query("""
    SELECT c.customer_id FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
    WHERE o.order_id IS NULL
""", conn)
check("A-SQL-Q4: 25 customers", len(q4) == 25, 25, len(q4))

q5 = pd.read_sql_query("""
    WITH cs AS (SELECT customer_id, SUM(order_amount) AS ts FROM orders WHERE status='completed' GROUP BY customer_id),
    av AS (SELECT AVG(ts) AS avg_t FROM cs)
    SELECT cs.customer_id FROM cs, av WHERE cs.ts > av.avg_t
""", conn)
check("A-SQL-Q5: 550 customers", len(q5) == 550, 550, len(q5))

q7 = pd.read_sql_query("""
    SELECT SUM(revenue) OVER (ORDER BY day) AS cr
    FROM (SELECT SUBSTR(order_ts,1,10) AS day, SUM(order_amount) AS revenue
          FROM orders WHERE status='completed' GROUP BY SUBSTR(order_ts,1,10))
    ORDER BY day
""", conn)
check("A-SQL-Q7: 183 days", len(q7) == 183, 183, len(q7))
check("A-SQL-Q7: last cumul", round(q7['cr'].iloc[-1], 2) == 2998668.80, 2998668.80, round(q7['cr'].iloc[-1], 2))

conn.close()

# ============================================================
# PART B CHECKPOINTS
# ============================================================
print("\n--- Part B Checks ---")

segments = pd.read_csv("outputs/customer_segments.csv")
check("B1: 1175 RFM customers", len(segments) == 1175, 1175, len(segments))
check("B1a: recency median 19", segments['recency'].median() == 19, 19, segments['recency'].median())
check("B1b: recency min 0", segments['recency'].min() == 0, 0, segments['recency'].min())
check("B1c: recency max 180", segments['recency'].max() == 180, 180, segments['recency'].max())
check("B1d: monetary mean ~2552", abs(segments['monetary'].mean() - 2552.06) < 1, 2552.06, round(segments['monetary'].mean(), 2))

# Segment columns
expected_cols = {'customer_id', 'recency', 'frequency', 'monetary', 'segment_id', 'segment_name', 'is_premium'}
check("B3: segment columns", set(segments.columns) == expected_cols, expected_cols, set(segments.columns))

# Premium segment exists
check("B3a: has Premium segment", 'Premium' in segments['segment_name'].values, True, 'Premium' in segments['segment_name'].values)

# Lapse model exists
check("B8: lapse model file", os.path.exists("outputs/lapse_model.joblib"), True, os.path.exists("outputs/lapse_model.joblib"))

# Retention list exists
check("B8b: retention list file", os.path.exists("outputs/retention_list.csv"), True, os.path.exists("outputs/retention_list.csv"))
retention = pd.read_csv("outputs/retention_list.csv")
ret_cols = {'customer_id', 'lapse_probability', 'actually_lapsed'}
check("B8c: retention columns", set(retention.columns) == ret_cols, ret_cols, set(retention.columns))

# ============================================================
# PART C CHECKPOINTS
# ============================================================
print("\n--- Part C Checks ---")

triage = pd.read_csv("outputs/triage_results.csv")
check("C7: triage 164 rows", len(triage) == 164, 164, len(triage))

budget = pd.read_csv("outputs/triage_budget_demo.csv")
budget_exhausted = (budget['status'] == 'budget_exhausted').sum()
check("C7a: at least 1 budget_exhausted", budget_exhausted >= 1, ">=1", budget_exhausted)

# ============================================================
# PART D CHECKPOINTS
# ============================================================
print("\n--- Part D Checks ---")

# agent_runs.jsonl exists with lines
check("D6: agent_runs.jsonl exists", os.path.exists("outputs/agent_runs.jsonl"), True, os.path.exists("outputs/agent_runs.jsonl"))
with open("outputs/agent_runs.jsonl", "r") as f:
    agent_lines = [json.loads(line) for line in f if line.strip()]
check("D6a: >= 7 agent run lines", len(agent_lines) >= 7, ">=7", len(agent_lines))

# Check S1 result
s1_runs = [r for r in agent_lines if r.get('scenario_id') == 'S1']
if s1_runs:
    s1 = s1_runs[0]
    check("D6-S1: status REFUND", s1['status'] == 'REFUND', 'REFUND', s1['status'])
    check("D6-S1: 5 LLM calls", s1['llm_calls'] == 5, 5, s1['llm_calls'])
    check("D6-S1: answer contains 128.25", '128.25' in s1['final_answer'], True, '128.25' in s1['final_answer'])

# Check S7
s7_runs = [r for r in agent_lines if r.get('scenario_id') == 'S7']
if s7_runs:
    s7 = s7_runs[0]
    check("D6-S7: FALLBACK_MAX_ITERATIONS", s7['status'] == 'FALLBACK_MAX_ITERATIONS', 'FALLBACK_MAX_ITERATIONS', s7['status'])
    check("D6-S7: 6 LLM calls", s7['llm_calls'] == 6, 6, s7['llm_calls'])
    check("D6-S7: tokens < 6000", s7['tokens_used'] < 6000, True, s7['tokens_used'])

# refund_ledger.jsonl exists
check("D5: refund_ledger.jsonl exists", os.path.exists("outputs/refund_ledger.jsonl"), True, os.path.exists("outputs/refund_ledger.jsonl"))

# state/ directory exists with checkpoints
check("D8: state/ exists", os.path.isdir("state"), True, os.path.isdir("state"))
state_files = os.listdir("state")
check("D8a: checkpoint files exist", len(state_files) > 0, ">0", len(state_files))

# Check required file paths
print("\n--- File Existence Checks ---")
required_files = [
    "generate_data.py", "mock_llm.py", "requirements.txt", ".gitignore", "README.md",
    "data/clean/customers.csv", "data/clean/products.csv", "data/clean/orders.csv",
    "data/instamart.db", "sql/queries.sql",
    "outputs/customer_segments.csv", "outputs/lapse_model.joblib", "outputs/retention_list.csv",
    "outputs/triage_results.csv", "outputs/triage_budget_demo.csv",
    "outputs/agent_runs.jsonl", "outputs/refund_ledger.jsonl"
]
for fp in required_files:
    exists = os.path.exists(fp)
    check(f"File: {fp}", exists, True, exists)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print(f"{'Check':<50} {'Expected':<15} {'Actual':<15} {'Status'}")
print("=" * 70)
for r in results:
    print(f"{r['check']:<50} {r['expected']:<15} {r['actual']:<15} {r['status']}")

n_pass = sum(1 for r in results if r['status'] == 'PASS')
n_fail = sum(1 for r in results if r['status'] == 'FAIL')
print(f"\nTotal: {n_pass} PASS, {n_fail} FAIL out of {len(results)} checks")

if n_fail > 0:
    print("\n*** SOME CHECKS FAILED ***")
    sys.exit(1)
else:
    print("\n*** ALL CHECKS PASSED ***")
    sys.exit(0)
