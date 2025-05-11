# feature_engineering_pipeline.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

# --- Feature Generators ---
class FilterTopLocations(BaseEstimator, TransformerMixin):
    """
    Keeps only records for specified top-N locations.
    """
    def __init__(self, top_locations):
        self.top_locations = top_locations

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[X['location_id'].isin(self.top_locations)].copy()

class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extracts hour, day of week, and month from a datetime column.
    """
    def __init__(self, datetime_column='datetime'):
        self.datetime_column = datetime_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        dt = pd.to_datetime(X_[self.datetime_column])
        X_['hour'] = dt.dt.hour
        X_['dayofweek'] = dt.dt.dayofweek
        X_['month'] = dt.dt.month
        return X_

class LagFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generates lag features for each location-level time series.
    """
    def __init__(self, n_lags=28, group_column='location_id', datetime_column='datetime', target_column='trip_count'):
        self.n_lags = n_lags
        self.group_column = group_column
        self.datetime_column = datetime_column
        self.target_column = target_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_.sort_values(by=[self.group_column, self.datetime_column], inplace=True)
        for lag in range(1, self.n_lags + 1):
            X_[f'lag_{lag}'] = X_.groupby(self.group_column)[self.target_column].shift(lag)
        return X_.dropna().reset_index(drop=True)

# --- Pipeline Builder ---
def build_feature_pipeline(top_locations, n_lags=28):
    feature_steps = [
        ('filter', FilterTopLocations(top_locations)),
        ('dt', DateTimeFeatures('datetime')),
        ('lag', LagFeatureGenerator(n_lags, 'location_id', 'datetime', 'trip_count'))
    ]
    feature_pipeline = Pipeline(feature_steps)

    datetime_feats = ['hour', 'dayofweek', 'month']
    lag_feats = [f'lag_{i}' for i in range(1, n_lags+1)]
    numeric_features = datetime_feats + lag_feats
    categorical_features = ['location_id']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    return Pipeline([('features', feature_pipeline), ('preprocess', preprocessor)])

# --- Main Script ---
if __name__ == '__main__':
    import os

    # Load the cleaned Parquet file
    input_path = '/Users/kaushalshivaprakash/Desktop/project3/data/processed/cleaned_citibike/citibike_2023_top3.parquet'
    df = pd.read_parquet(input_path)

    # Rename and parse
    df = df.rename(columns={'started_at':'datetime', 'start_station_id':'location_id'})
    df['datetime'] = pd.to_datetime(df['datetime']).dt.floor('H')

    # Aggregate to hourly trip counts
    df_agg = (
        df.groupby(['location_id', 'datetime']).size()
          .reset_index(name='trip_count')
    )

    # Top 3 stations
    top_locs = df_agg['location_id'].value_counts().nlargest(3).index.tolist()

    # Train/test split: last 30 days as test
    max_dt = df_agg['datetime'].max()
    cutoff = max_dt - pd.Timedelta(days=30)
    train = df_agg[df_agg['datetime'] < cutoff]
    test  = df_agg[df_agg['datetime'] >= cutoff]

    if train.empty or test.empty:
        raise ValueError(
            f"Train or test split is empty! Train size: {len(train)}, Test size: {len(test)}"
        )

    # Build, fit, and transform
    pipeline = build_feature_pipeline(top_locs, n_lags=28)
    X_train = pipeline.fit_transform(train)
    X_test  = pipeline.transform(test)

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    # Save pipeline
    model_path = 'models/feature_pipeline.pkl'
    joblib.dump(pipeline, model_path)

    # Confirmation messages
    print(f'Pipeline built. Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}')
    print(f'Feature engineering pipeline successfully saved to {model_path}')
