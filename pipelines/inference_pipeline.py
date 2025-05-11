# inference_pipeline.py
"""
Standalone batch inference script reading Parquet input directly.
"""

import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline as SklearnPipeline

# ── 1) CONFIGURE YOUR PATHS HERE ────────────────────────────────────────
FEATURE_PIPELINE = "/Users/kaushalshivaprakash/Desktop/project3/pipelines/models/feature_pipeline.pkl"
MODEL_PATH       = "/Users/kaushalshivaprakash/Desktop/project3/pipelines/models/best_model.pkl"
INPUT_PARQUET    = "/Users/kaushalshivaprakash/Desktop/project3/data/processed/cleaned_citibike/citibike_2023_top3.parquet"
OUTPUT_CSV       = "/Users/kaushalshivaprakash/Desktop/project3/pipelines/output/predictions.csv"

# ── 2) LOAD & COMPOSE ──────────────────────────────────────────────────
def load_pipeline(pipeline_path: str, model_path: str) -> SklearnPipeline:
    preprocessor = joblib.load(pipeline_path)
    model        = joblib.load(model_path)
    return SklearnPipeline([
        ("preprocessing", preprocessor),
        ("estimator",     model)
    ])

# ── 3) BATCH INFERENCE ─────────────────────────────────────────────────
def run_batch_inference(pipeline: SklearnPipeline, input_path: str, output_path: str):
    # Read Parquet input
    df = pd.read_parquet(input_path)
    # Ensure timestamp is datetime
    df["started_at"] = pd.to_datetime(df["started_at"])
    # Bucket by hour (must match training)
    df["hour_bucket"] = df["started_at"].dt.floor("H")
    # Prepare feature frame
    X = pd.DataFrame({
        "start_station_id": df["start_station_id"],
        "hour_bucket":      df["hour_bucket"]
    })
    # Run predictions
    df["predicted_trips"] = pipeline.predict(X)
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Write out full DataFrame with predictions
    df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")

# ── 4) MAIN EXECUTION ──────────────────────────────────────────────────
def main():
    # Validate existence of all paths
    for path in (FEATURE_PIPELINE, MODEL_PATH, INPUT_PARQUET):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
    # Load the combined pipeline
    pipeline = load_pipeline(FEATURE_PIPELINE, MODEL_PATH)
    # Run and save predictions
    run_batch_inference(pipeline, INPUT_PARQUET, OUTPUT_CSV)
    # Final confirmation
    print("🚀 Inference pipeline created and executed successfully!")

if __name__ == "__main__":
    main()
