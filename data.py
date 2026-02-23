import os
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch
from dotenv import load_dotenv

load_dotenv()

SOCRATA_DATA_URL = "https://data.cityofnewyork.us/resource/t29m-gskq.json" # 2020 Yellow Taxi Data as an example

@st.cache_data
def fetch_data(limit=5000):
    """
    Fetches NYC Yellow Taxi data from Socrata API.
    Uses SOCRATA_APP_TOKEN from environment if available.
    """
    url = f"{SOCRATA_DATA_URL}?$limit={limit}"
    
    headers = {}
    app_token = os.environ.get("SOCRATA_APP_TOKEN")
    if app_token:
        headers["X-App-Token"] = app_token

    print(f"Fetching {limit} rows from {url}...")
    
    if headers:
        import requests
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
    else:
        df = pd.read_json(url)
        
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans invalid rows and extracts temporal features.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Ensure datetime columns are parsed
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    
    # Drop rows with invalid or nonsensical data (often present in TLC data)
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
