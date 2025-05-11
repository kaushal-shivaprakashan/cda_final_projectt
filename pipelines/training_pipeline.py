# training_pipeline.py
"""
Standalone model training pipeline.
Loads raw data, aggregates rides into hourly counts,
trains a LightGBM model with preprocessing pipeline,
saves artifacts to disk, and prints confirmation.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from joblib import dump

# ── 1) CONFIGURATION ─────────────────────────────────────────────────
PARQUET_PATH = "/Users/kaushalshivaprakash/Desktop/project3/data/processed/cleaned_citibike/citibike_2023_top3.parquet"
MODEL_DIR    = "/Users/kaushalshivaprakash/Desktop/project3/pipelines/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 2) DATA LOADING & AGGREGATION ─────────────────────────────────────
# Read ride-level Parquet data
df = pd.read_parquet(PARQUET_PATH)
df["started_at"] = pd.to_datetime(df["started_at"])
# Bucket into hourly periods
df["hour_bucket"] = df["started_at"].dt.floor("H")
# Aggregate trips per station per hour
agg = (
    df.groupby(["start_station_id", "hour_bucket"])  
      .agg(target_trips=("ride_id", "count"))
      .reset_index()
)

# ── 3) Prepare features and target ─────────────────────────────────────
X = agg[["start_station_id", "hour_bucket"]]
y = agg["target_trips"]

# ── 4) Define preprocessing pipeline ──────────────────────────────────
def select_hour_bucket(df):
    return df[["hour_bucket"]]

def extract_datetime_parts(df):
    return pd.DataFrame({
        "year":  df["hour_bucket"].dt.year,
        "month": df["hour_bucket"].dt.month,
        "day":   df["hour_bucket"].dt.day,
        "hour":  df["hour_bucket"].dt.hour
    })

def select_station_id(df):
    return df[["start_station_id"]]

datetime_pipeline = Pipeline([
    ("select_dt", FunctionTransformer(select_hour_bucket, validate=False)),
    ("extract",   FunctionTransformer(extract_datetime_parts, validate=False))
])

station_pipeline = Pipeline([
    ("select_loc", FunctionTransformer(select_station_id, validate=False)),
    ("onehot",     OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("dt_feats",  datetime_pipeline,    ["hour_bucket"]),
    ("loc_feats", station_pipeline,     ["start_station_id"]),
])

# ── 5) Create full modeling pipeline ──────────────────────────────────
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("estimator",    LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    ))
])

# ── 6) Train/test split & fit ─────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# ── 7) Evaluation ─────────────────────────────────────────────────────
val_preds = pipeline.predict(X_val)
mae = abs(val_preds - y_val).mean()
print(f"Validation MAE: {mae:.3f}")

# ── 8) Save artifacts ─────────────────────────────────────────────────
feature_pipeline_path = os.path.join(MODEL_DIR, "feature_pipeline.pkl")
# Save model under the name 'model_training_pipeline.pkl'
model_path            = os.path.join(MODEL_DIR, "model_training_pipeline.pkl")

dump(preprocessor, feature_pipeline_path)
print(f"✔ Saved feature pipeline -> {feature_pipeline_path}")

dump(pipeline.named_steps["estimator"], model_path)
print(f"✔ Saved model            -> {model_path}")

# ── 9) Final confirmation ─────────────────────────────────────────────
print("🚀 Model training pipeline executed successfully and saved as 'model_training_pipeline.pkl'!")
