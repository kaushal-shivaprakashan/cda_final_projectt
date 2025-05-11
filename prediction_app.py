import os
import streamlit as st
import pandas as pd
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit App: Prediction + Monitoring
st.set_page_config(page_title="Citi Bike Dashboard", layout="wide")
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"

# MLflow configuration
# Ensure MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD are set in your environment if needed
mlflow.set_tracking_uri("https://dagshub.com/kaushal-shivaprakashan/final_project.mlflow")
EXPERIMENT_NAME = "CitiBike_Remote_Experiment"

# Navigation tabs
tabs = st.tabs(["Prediction", "Monitoring"])

# ==================== Prediction Tab ====================
with tabs[0]:
    st.header("🛣️ Hourly Trip Prediction")

    @st.cache_resource
    def load_assets():
        pipe_path = MODEL_DIR / "feature_pipeline.pkl"
        model_path = MODEL_DIR / "best_model.pkl"
        if not pipe_path.exists() or not model_path.exists():
            st.error("Missing model artifacts in 'models/' directory.")
            st.stop()
        pipeline = joblib.load(pipe_path)
        model = joblib.load(model_path)
        return pipeline, model

    pipeline, model = load_assets()

    # Sidebar inputs
    st.sidebar.subheader("Prediction Inputs")
    date = st.sidebar.date_input("Pickup Date", pd.to_datetime("2023-01-01"))
    hour = st.sidebar.slider("Pickup Hour", 0, 23, 12)
    location = st.sidebar.selectbox("Location ID", [101, 102, 103])  # update IDs if needed

    # Prepare input and predict
    input_df = pd.DataFrame([{
        "location_id": location,
        "pickup_date": date,
        "pickup_hour": hour
    }])
    features = pipeline.transform(input_df)
    pred = model.predict(features)[0]

    st.metric(label="Predicted Trips per Hour", value=f"{pred:.2f}")

# ==================== Monitoring Tab ====================
with tabs[1]:
    st.header("📈 Model Performance Monitoring")

    @st.cache_data(ttl=300)
    def fetch_metrics():
        client = MlflowClient()
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        if not exp:
            st.error(f"Experiment '{EXPERIMENT_NAME}' not found.")
            st.stop()
        runs = client.search_runs(exp.experiment_id, order_by=["attributes.start_time ASC"])
        records = []
        for run in runs:
            hist = client.get_metric_history(run.info.run_id, "mae")
            if hist:
                records.append({
                    "time": run.info.start_time,
                    "run_id": run.info.run_id,
                    "mae": hist[-1].value
                })
        df = pd.DataFrame(records)
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df.set_index("time").sort_index()

    metrics_df = fetch_metrics()

    if not metrics_df.empty:
        st.subheader("Mean Absolute Error over Time")
        st.line_chart(metrics_df["mae"])
        st.subheader("Runs Detail")
        st.dataframe(metrics_df)
    else:
        st.info("Awaiting logged runs to display metrics.")
