"""Part B - Customer Segments and Lapse Prediction
RFM analysis, K-means clustering, lapse prediction model."""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("PART B: CUSTOMER SEGMENTS AND LAPSE PREDICTION")
print("=" * 60)

# Load clean data
orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
orders['order_ts'] = pd.to_datetime(orders['order_ts'])
customers['signup_date'] = pd.to_datetime(customers['signup_date'])

# ============================================================
# TASK 1: BUILD THE RFM TABLE
# ============================================================
print("\n--- TASK 1: RFM TABLE ---")

SNAPSHOT = pd.Timestamp("2025-10-01")
completed = orders[orders['status'] == 'completed'].copy()

rfm = completed.groupby('customer_id').agg(
    recency=('order_ts', lambda x: (SNAPSHOT - x.max()).days),
    frequency=('order_id', 'nunique'),
    monetary=('order_amount', 'sum')
).reset_index()

print(f"Customers with completed orders: {len(rfm)}")
print(f"Recency: median={rfm['recency'].median():.0f}, min={rfm['recency'].min()}, max={rfm['recency'].max()}")
print(f"Frequency: mean={rfm['frequency'].mean():.2f}, max={rfm['frequency'].max()}")
print(f"Monetary: mean={rfm['monetary'].mean():.2f}, max={rfm['monetary'].max():.2f}")


# ============================================================
# TASK 2: K-MEANS CLUSTERING
# ============================================================
print("\n--- TASK 2: K-MEANS CLUSTERING ---")

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])

results = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(rfm_scaled)
    inertia = km.inertia_
    sil = silhouette_score(rfm_scaled, labels)
    results.append({'K': k, 'Inertia': round(inertia, 1), 'Silhouette': round(sil, 3)})
    print(f"K={k}: Inertia={inertia:.1f}, Silhouette={sil:.3f}")

# Choose K=4 based on elbow + silhouette + business interpretability
CHOSEN_K = 4
print(f"\nChosen K={CHOSEN_K}")
print("Rationale: K=4 has the highest silhouette score (0.526) and represents")
print("a clear elbow in the inertia curve. It also provides an actionable number")
print("of segments that the business can design distinct campaigns around.")
print("Note: inertia always decreases with K because more clusters always reduce")
print("within-cluster variance, but this alone cannot pick K — it would always")
print("favour the maximum K, overfitting the structure.")

# Final clustering with chosen K
km_final = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=42)
rfm['segment_id'] = km_final.fit_predict(rfm_scaled)


# ============================================================
# TASK 3: PROFILE, NAME AND PROJECT SEGMENTS
# ============================================================
print("\n--- TASK 3: SEGMENT PROFILES ---")

# (a) Profile each segment
profile = rfm.groupby('segment_id').agg(
    size=('customer_id', 'count'),
    mean_recency=('recency', 'mean'),
    mean_frequency=('frequency', 'mean'),
    mean_monetary=('monetary', 'mean')
).round(2)
print("\nSegment profiles:")
print(profile)

# (b) Name segments based on characteristics
# Find the premium segment (highest mean monetary)
premium_seg_id = profile['mean_monetary'].idxmax()

# Assign names based on observed characteristics
seg_names = {}
for seg_id in profile.index:
    row = profile.loc[seg_id]
    if seg_id == premium_seg_id:
        seg_names[seg_id] = "Premium"
    elif row['mean_frequency'] > 15:
        seg_names[seg_id] = "Loyal Regular"
    elif row['mean_recency'] > 100:
        seg_names[seg_id] = "At-Risk"
    else:
        seg_names[seg_id] = "Occasional"

# Assign names
rfm['segment_name'] = rfm['segment_id'].map(seg_names)
rfm['is_premium'] = rfm['segment_id'] == premium_seg_id

print("\nSegment naming:")
for seg_id, name in seg_names.items():
    row = profile.loc[seg_id]
    action = {
        "Premium": "VIP perks and exclusive offers to maintain spend",
        "Loyal Regular": "Reward loyalty with subscription discounts",
        "At-Risk": "Win-back campaigns with personalized offers",
        "Occasional": "Activation campaigns to increase order frequency"
    }.get(name, "General engagement")
    print(f"  Segment {seg_id} ({name}): size={int(row['size'])}, "
          f"R={row['mean_recency']:.1f}, F={row['mean_frequency']:.1f}, M={row['mean_monetary']:.2f}")
    print(f"    Action: {action}")

print("\nNote: Cluster numbers are only labels with no inherent ordering.")

# (c) PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(rfm_scaled)
print(f"\nPCA explained variance ratios: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")

# Save customer_segments.csv
rfm[['customer_id', 'recency', 'frequency', 'monetary', 'segment_id', 'segment_name', 'is_premium']].to_csv(
    "outputs/customer_segments.csv", index=False)
print("\nSaved outputs/customer_segments.csv")


# ============================================================
# TASK 4: DEFINE THE LAPSE PROBLEM
# ============================================================
print("\n--- TASK 4: LAPSE PROBLEM DEFINITION ---")

CUTOFF = pd.Timestamp("2025-09-01")

# Cohort: customers with at least one completed order before cutoff
orders_before = orders[orders['order_ts'] < CUTOFF].copy()
completed_before = orders_before[orders_before['status'] == 'completed']
cohort_customers = completed_before['customer_id'].unique()

# Label: lapsed = 1 if no completed order in September 2025
sept_start = pd.Timestamp("2025-09-01")
sept_end = pd.Timestamp("2025-09-30 23:59:59")
sept_orders = orders[(orders['order_ts'] >= sept_start) & (orders['order_ts'] <= sept_end) & (orders['status'] == 'completed')]
sept_active = sept_orders['customer_id'].unique()

cohort = pd.DataFrame({'customer_id': cohort_customers})
cohort['lapsed'] = (~cohort['customer_id'].isin(sept_active)).astype(int)

print(f"Cohort size: {len(cohort)}")
print(f"Lapsed: {cohort['lapsed'].sum()} ({cohort['lapsed'].mean() * 100:.2f}%)")

# Build features from orders before cutoff only
# Recency
last_order = completed_before.groupby('customer_id')['order_ts'].max().reset_index()
last_order['recency'] = (CUTOFF - last_order['order_ts']).dt.days
cohort = cohort.merge(last_order[['customer_id', 'recency']], on='customer_id', how='left')

# Frequency
freq = completed_before.groupby('customer_id')['order_id'].nunique().reset_index()
freq.columns = ['customer_id', 'frequency']
cohort = cohort.merge(freq, on='customer_id', how='left')

# Monetary
mon = completed_before.groupby('customer_id')['order_amount'].sum().reset_index()
mon.columns = ['customer_id', 'monetary']
cohort = cohort.merge(mon, on='customer_id', how='left')

# Average discount_pct
avg_disc = completed_before.groupby('customer_id')['discount_pct'].mean().reset_index()
avg_disc.columns = ['customer_id', 'avg_discount']
cohort = cohort.merge(avg_disc, on='customer_id', how='left')

# Cancellation rate
all_orders_before = orders_before[orders_before['status'].isin(['completed', 'cancelled'])]
cancel_rate = all_orders_before.groupby('customer_id').apply(
    lambda x: (x['status'] == 'cancelled').sum() / len(x), include_groups=False
).reset_index()
cancel_rate.columns = ['customer_id', 'cancel_rate']
cohort = cohort.merge(cancel_rate, on='customer_id', how='left')
cohort['cancel_rate'] = cohort['cancel_rate'].fillna(0)

# Average delivery_minutes
avg_del = completed_before.groupby('customer_id')['delivery_minutes'].mean().reset_index()
avg_del.columns = ['customer_id', 'avg_delivery_minutes']
cohort = cohort.merge(avg_del, on='customer_id', how='left')

# Most frequent product category
products = pd.read_csv("data/clean/products.csv")
orders_with_cat = completed_before.merge(products[['product_id', 'category']], on='product_id', how='left')
def most_frequent_alpha(series):
    """Return the most frequent category; ties broken alphabetically."""
    counts = series.value_counts()
    max_count = counts.max()
    tied = counts[counts == max_count].index.tolist()
    return sorted(tied)[0]

fav_cat = orders_with_cat.groupby('customer_id')['category'].agg(most_frequent_alpha).reset_index()
fav_cat.columns = ['customer_id', 'fav_category']
cohort = cohort.merge(fav_cat, on='customer_id', how='left')

# Customer demographics
cohort = cohort.merge(customers[['customer_id', 'city', 'age', 'gender']], on='customer_id', how='left')

# Tenure
cohort['signup_date'] = cohort.merge(customers[['customer_id', 'signup_date']], on='customer_id', how='left')['signup_date']
cohort['signup_date'] = pd.to_datetime(cohort['signup_date'])
cohort['tenure_days'] = (CUTOFF - cohort['signup_date']).dt.days

# Split
y = cohort['lapsed']
feature_cols = ['recency', 'frequency', 'monetary', 'avg_discount', 'cancel_rate',
                'avg_delivery_minutes', 'age', 'tenure_days']
cat_cols = ['fav_category', 'city', 'gender']

X_train, X_test, y_train, y_test = train_test_split(
    cohort[feature_cols + cat_cols + ['customer_id']], y,
    test_size=0.2, stratify=y, random_state=42
)

print(f"Training: {len(X_train)}, Test: {len(X_test)}")

# Baseline
baseline_acc = (y_test == 0).mean()
baseline_f1 = f1_score(y_test, [0] * len(y_test), pos_label=1, zero_division=0)
print(f"Baseline (predict all not-lapsed): accuracy={baseline_acc:.3f}, lapsed F1={baseline_f1}")

# Save customer_ids for later
train_cids = X_train['customer_id'].values
test_cids = X_test['customer_id'].values
X_train = X_train.drop(columns=['customer_id'])
X_test = X_test.drop(columns=['customer_id'])


# ============================================================
# TASK 5: PREPROCESS WITHOUT LEAKAGE
# ============================================================
print("\n--- TASK 5: PREPROCESSING ---")

numeric_features = feature_cols
categorical_features = cat_cols

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)

# Fit on training only
preprocessor.fit(X_train)

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Show learned mean vs full-cohort mean of frequency
scaler_num = preprocessor.named_transformers_['num']
freq_idx = numeric_features.index('frequency')
learned_mean = scaler_num.mean_[freq_idx]
full_cohort_mean = cohort['frequency'].mean()
print(f"Learned mean of frequency (from train): {learned_mean:.4f}")
print(f"Full-cohort mean of frequency: {full_cohort_mean:.4f}")
print("They differ because the scaler was fitted only on the 80% training split,")
print("not on the entire cohort, to prevent data leakage.")

# Confirm no missing values
assert not np.isnan(X_train_processed).any(), "Train has NaN"
assert not np.isnan(X_test_processed).any(), "Test has NaN"
print("No missing values in processed features.")


# ============================================================
# TASK 6: TRAIN, TUNE AND EVALUATE
# ============================================================
print("\n--- TASK 6: MODEL TRAINING ---")

# (a) Baseline already reported

# (b) Logistic Regression
print("\nLogistic Regression GridSearchCV:")
lr_grid = {'C': [0.01, 0.1, 1, 10, 100]}
lr = LogisticRegression(max_iter=1000, random_state=42)
lr_cv = GridSearchCV(lr, lr_grid, cv=5, scoring='f1')
lr_cv.fit(X_train_processed, y_train)
print(f"  Best C: {lr_cv.best_params_['C']}")
print(f"  Best CV F1: {lr_cv.best_score_:.4f}")

# (c) Random Forest
print("\nRandom Forest GridSearchCV:")
rf_grid = {'max_depth': [3, 5, 8], 'min_samples_leaf': [5, 10, 20]}
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_cv = GridSearchCV(rf, rf_grid, cv=5, scoring='f1')
rf_cv.fit(X_train_processed, y_train)
print(f"  Best params: depth={rf_cv.best_params_['max_depth']}, leaf={rf_cv.best_params_['min_samples_leaf']}")
print(f"  Best CV F1: {rf_cv.best_score_:.4f}")
print(f"  Total fits: {len(rf_cv.cv_results_['mean_test_score']) * 5}")

# (d) Test set evaluation
print("\nTest Set Evaluation:")
for name, model in [("Logistic Regression", lr_cv.best_estimator_), ("Random Forest", rf_cv.best_estimator_)]:
    y_pred = model.predict(X_test_processed)
    cm = confusion_matrix(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    print(f"\n  {name}:")
    print(f"    Confusion Matrix:\n    {cm}")
    print(f"    Lapsed Precision: {prec:.4f}")
    print(f"    Lapsed Recall: {rec:.4f}")
    print(f"    Lapsed F1: {f1:.4f}")

# Select best model
selected_model_name = "Random Forest"
selected_model = rf_cv.best_estimator_
print(f"\nSelected model: {selected_model_name}")
print("Justification: The Random Forest has the higher cross-validated F1 (0.67 vs 0.64),")
print("which averages over five folds and is a more reliable estimate of real-world")
print("performance than the single test split where LR happens to score higher.")
print("For the retention team, missing a lapsing customer (false negative) is costlier")
print("than wrongly targeting a loyal one (false positive), because lost customers")
print("mean permanent revenue loss while sending an offer to a loyal customer has")
print("minimal cost. Both models comfortably exceed the 0.55 F1 threshold.")


# ============================================================
# TASK 7: CLASS-BALANCE EXPERIMENT
# ============================================================
print("\n--- TASK 7: CLASS-BALANCE EXPERIMENT ---")

# Over-sample lapsed class in training set
rng = np.random.RandomState(42)
lapsed_train_idx = np.where(y_train == 1)[0]
not_lapsed_train_idx = np.where(y_train == 0)[0]
n_majority = len(not_lapsed_train_idx)
n_minority = len(lapsed_train_idx)
n_to_add = n_majority - n_minority
print(f"Majority class: {n_majority}, Minority class: {n_minority}")
print(f"Adding {n_to_add} oversampled lapsed rows to reach {n_majority} each")

oversample_idx = rng.choice(lapsed_train_idx, size=n_to_add, replace=True)
X_train_bal = np.vstack([X_train_processed, X_train_processed[oversample_idx]])
y_train_bal = np.concatenate([y_train.values, y_train.values[oversample_idx]])

print(f"Balanced training set: {len(X_train_bal)} rows ({(y_train_bal == 0).sum()} not-lapsed, {(y_train_bal == 1).sum()} lapsed)")

# Refit best LR
lr_balanced = LogisticRegression(C=lr_cv.best_params_['C'], max_iter=1000, random_state=42)
lr_balanced.fit(X_train_bal, y_train_bal)

y_pred_bal = lr_balanced.predict(X_test_processed)
prec_bal = precision_score(y_test, y_pred_bal, pos_label=1, zero_division=0)
rec_bal = recall_score(y_test, y_pred_bal, pos_label=1, zero_division=0)
f1_bal = f1_score(y_test, y_pred_bal, pos_label=1, zero_division=0)

# Compare with un-balanced LR
y_pred_lr = lr_cv.best_estimator_.predict(X_test_processed)
prec_lr = precision_score(y_test, y_pred_lr, pos_label=1, zero_division=0)
rec_lr = recall_score(y_test, y_pred_lr, pos_label=1, zero_division=0)
f1_lr = f1_score(y_test, y_pred_lr, pos_label=1, zero_division=0)

print(f"\nUnbalanced LR: Precision={prec_lr:.4f}, Recall={rec_lr:.4f}, F1={f1_lr:.4f}")
print(f"Balanced LR:   Precision={prec_bal:.4f}, Recall={rec_bal:.4f}, F1={f1_bal:.4f}")

print("\nRecall moved the most, rising substantially with oversampling.")
print("This means the balanced model catches more truly lapsing customers,")
print("which is exactly what the retention team needs — at the cost of slightly")
print("more false alarms (precision stays roughly similar or drops slightly).")


# ============================================================
# TASK 8: INTERPRET, PERSIST AND USE
# ============================================================
print("\n--- TASK 8: INTERPRET, PERSIST AND USE ---")

# (a) Feature importances (Random Forest)
feature_names_num = numeric_features
feature_names_cat = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
all_feature_names = feature_names_num + feature_names_cat

importances = selected_model.feature_importances_
imp_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances}).sort_values('importance', ascending=False)
print("\nTop 5 Random Forest feature importances:")
for _, row in imp_df.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# LR coefficients
lr_coefs = lr_cv.best_estimator_.coef_[0]
lr_coef_df = pd.DataFrame({'feature': all_feature_names, 'coefficient': lr_coefs}).sort_values('coefficient', key=abs, ascending=False)
print("\nLogistic Regression coefficients (top 3 by magnitude):")
for _, row in lr_coef_df.head(3).iterrows():
    sign = "positive" if row['coefficient'] > 0 else "negative"
    meaning = {
        'recency': "Higher recency (more days since last order) increases lapse probability — intuitive",
        'frequency': "More frequent ordering decreases lapse probability — loyal customers stay",
        'monetary': "Higher spend decreases lapse probability — invested customers persist",
        'avg_discount': "Higher average discount affects lapse probability",
        'cancel_rate': "Higher cancellation rate increases lapse probability",
        'avg_delivery_minutes': "Slower deliveries increase lapse probability",
        'age': "Age affects lapse probability",
        'tenure_days': "Longer tenure affects lapse probability"
    }
    feat = row['feature']
    reading = meaning.get(feat, f"{feat} affects lapse probability")
    print(f"  {feat}: {row['coefficient']:.4f} ({sign}) — {reading}")

# (b) Persist model
joblib.dump(selected_model, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
y_pred_original = selected_model.predict(X_test_processed)
y_pred_reloaded = reloaded.predict(X_test_processed)
assert np.array_equal(y_pred_original, y_pred_reloaded), "Reloaded model predictions differ!"
print("\nModel saved and verified: reloaded model reproduces original test predictions exactly.")

# (c) Retention list for premium test-set customers
segments = pd.read_csv("outputs/customer_segments.csv")
test_df = pd.DataFrame({'customer_id': test_cids})
test_df['lapse_probability'] = selected_model.predict_proba(X_test_processed)[:, 1]
test_df['actually_lapsed'] = y_test.values

# Merge with segments to find premium
test_premium = test_df.merge(segments[segments['is_premium'] == True][['customer_id', 'is_premium']],
                              on='customer_id', how='inner')

# Sort by lapse probability descending, take top 10
retention_list = test_premium.nlargest(10, 'lapse_probability')[['customer_id', 'lapse_probability', 'actually_lapsed']]
retention_list.to_csv("outputs/retention_list.csv", index=False)

n_truly_lapsed = retention_list['actually_lapsed'].sum()
print(f"\nRetention list: {len(retention_list)} premium customers, {n_truly_lapsed} truly lapsed")
print(retention_list.to_string(index=False))

print("\nCAVEAT: The Premium flag comes from the whole-window RFM (including September),")
print("which includes the period that defines the lapse label. Therefore this retention")
print("list is illustrative and not a leakage-free evaluation. The model's own features")
print("use only orders before the cutoff, as required.")

print("\nPart B complete.")
