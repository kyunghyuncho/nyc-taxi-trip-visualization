import os
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch
from dotenv import load_dotenv

load_dotenv()

# TLC Trip Record Data (Yellow Taxi 2023-01)
TLC_PARQUET_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"

@st.cache_data
def fetch_data(limit=10000):
    """
    Fetches NYC Yellow Taxi data directly from the TLC Parquet S3 bucket.
    This is much faster than the JSON API.
    """
    print(f"Fetching {limit} rows from {TLC_PARQUET_URL}...")
    
    # We read the first `limit` rows. 
    # To do this efficiently without loading the whole file into memory first:
    try:
        df = pd.read_parquet(TLC_PARQUET_URL, engine='pyarrow')
        # Sample the dataframe to get a reasonable subset
        df = df.sample(n=min(limit, len(df)), random_state=42).reset_index(drop=True)
    except Exception as e:
        print(f"Failed to load parquet: {e}")
        return pd.DataFrame()
        
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans invalid rows and extracts temporal features.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Capitalization may vary in the parquet file
    pickup_col = 'tpep_pickup_datetime' if 'tpep_pickup_datetime' in df.columns else 'tpep_pickup_datetime'.capitalize()
    dropoff_col = 'tpep_dropoff_datetime' if 'tpep_dropoff_datetime' in df.columns else 'tpep_dropoff_datetime'.capitalize()

    if pickup_col not in df.columns and 'tpep_pickup_datetime' not in df.columns.str.lower():
         # Fallback search
         pickup_col = [c for c in df.columns if 'pickup' in c.lower() and 'datetime' in c.lower()][0]
         dropoff_col = [c for c in df.columns if 'dropoff' in c.lower() and 'datetime' in c.lower()][0]

    # Ensure datetime columns are parsed
    df[pickup_col] = pd.to_datetime(df[pickup_col])
    df[dropoff_col] = pd.to_datetime(df[dropoff_col])
    
    # Drop rows with invalid or nonsensical data (often present in TLC data)
    # The columns might be capitalized like 'Trip_distance'
    df.columns = df.columns.astype(str).str.lower()
    
    num_cols = ['trip_distance', 'fare_amount', 'total_amount']
    for col in num_cols:
        if col in df.columns:
             # convert to numeric, coerce errors to NaN
             df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove obvious outliers and negative distances/fares
    initial_len = len(df)
    df = df[
        (df['trip_distance'] > 0) & (df['trip_distance'] < 100) &
        (df['fare_amount'] > 0) & (df['fare_amount'] < 500) &
        (df['total_amount'] > 0)
    ]
    df = df.dropna()
    print(f"Dropped {initial_len - len(df)} invalid rows.")

    # Extract temporal features
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_dayofweek'] = df['tpep_pickup_datetime'].dt.dayofweek
    
    # Cast known categorical columns to string so Plotly maps them correctly 
    # instead of treating them as continuous colors
    categorical_columns = [
        'vendorid', 'ratecodeid', 'pulocationid', 'dolocationid', 
        'payment_type', 'pickup_dayofweek', 'pickup_hour', 'store_and_fwd_flag'
    ]
    for cat_col in categorical_columns:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str)
            
    return df


def transform_data(df: pd.DataFrame, feature_cols: list) -> tuple[torch.Tensor, StandardScaler]:
    """
    Applies One-Hot Encoding and StandardScaler to selected features.
    Returns the tensor and the scaler/transformer for potential inverse ops.
    """
    if not feature_cols:
        raise ValueError("No feature columns selected for transformation.")

    # Heuristic: if a column is an object/category, or specifically named as categorical, we OHE it
    # For TLC data, vendor_id, RatecodeID, payment_type are usually categorical.
    categorical_candidates = ['vendor_id', 'vendorid', 'ratecodeid', 'payment_type', 'pickup_dayofweek', 'pickup_hour']
    
    # identify which of the selected features are categorical vs numeric
    categorical_features = [col for col in feature_cols if col.lower() in categorical_candidates or df[col].dtype == 'object']
    numeric_features = [col for col in feature_cols if col not in categorical_features]

    # Preprocessors
    transformers = []
    if numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    # Fit and transform
    X_transformed = preprocessor.fit_transform(df[feature_cols])
    
    # Convert to PyTorch Tensor
    X_tensor = torch.tensor(X_transformed, dtype=torch.float32)

    return X_tensor, preprocessor
