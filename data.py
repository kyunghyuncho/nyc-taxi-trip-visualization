import os
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
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
    
    # NYC TLC Data Dictionary Mappings
    vendor_map = {1: 'Creative Mobile Technologies', 2: 'VeriFone Inc.'}
    ratecode_map = {
        1: 'Standard rate', 2: 'JFK', 3: 'Newark', 
        4: 'Nassau or Westchester', 5: 'Negotiated fare', 6: 'Group ride'
    }
    payment_map = {
        1: 'Credit card', 2: 'Cash', 3: 'No charge', 
        4: 'Dispute', 5: 'Unknown', 6: 'Voided trip'
    }
    dayofweek_map = {
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
        3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    }

    # Apply Mappings
    if 'vendorid' in df.columns:
        df['vendorid'] = pd.to_numeric(df['vendorid'], errors='coerce').map(vendor_map).fillna('Unknown')
    
    if 'ratecodeid' in df.columns:
        df['ratecodeid'] = pd.to_numeric(df['ratecodeid'], errors='coerce').map(ratecode_map).fillna('Unknown')
    
    if 'payment_type' in df.columns:
        df['payment_type'] = pd.to_numeric(df['payment_type'], errors='coerce').map(payment_map).fillna('Unknown')
        
    df['pickup_dayofweek'] = df['pickup_dayofweek'].map(dayofweek_map)

    # Map Location IDs to Boroughs and Zones
    try:
        lookup_df = pd.read_csv('https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv')
        borough_map = dict(zip(lookup_df['LocationID'], lookup_df['Borough']))
        zone_map = dict(zip(lookup_df['LocationID'], lookup_df['Zone']))
        
        if 'pulocationid' in df.columns:
            df['pu_borough'] = pd.to_numeric(df['pulocationid'], errors='coerce').map(borough_map).fillna('Unknown')
            df['pu_zone'] = pd.to_numeric(df['pulocationid'], errors='coerce').map(zone_map).fillna('Unknown')
            
        if 'dolocationid' in df.columns:
            df['do_borough'] = pd.to_numeric(df['dolocationid'], errors='coerce').map(borough_map).fillna('Unknown')
            df['do_zone'] = pd.to_numeric(df['dolocationid'], errors='coerce').map(zone_map).fillna('Unknown')
    except Exception as e:
        print(f"Failed to fetch or map taxi zones: {e}")

    # Cast other known categorical columns to string
    other_cat_cols = ['pulocationid', 'dolocationid', 'pickup_hour', 'store_and_fwd_flag', 'pu_borough', 'do_borough', 'pu_zone', 'do_zone']
    for cat_col in other_cat_cols:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str)
            
    return df


def transform_data(df: pd.DataFrame, feature_cols: list) -> tuple[torch.Tensor, torch.Tensor, StandardScaler]:
    """
    Splits the data into train/val sets.
    Applies One-Hot Encoding and StandardScaler to selected features.
    Returns the train tensor, val tensor, and the scaler/transformer.
    """
    if not feature_cols:
        raise ValueError("No feature columns selected for transformation.")

    # Heuristic: if a column is an object/category, or specifically named as categorical, we OHE it
    # For TLC data, vendor_id, RatecodeID, payment_type are usually categorical.
    categorical_candidates = ['vendor_id', 'vendorid', 'ratecodeid', 'payment_type', 'pickup_dayofweek', 'pickup_hour', 'pu_borough', 'do_borough', 'pu_zone', 'do_zone', 'pulocationid', 'dolocationid']

    
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

    # Split Data (80/20)
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

    # Fit and transform
    X_train_transformed = preprocessor.fit_transform(df_train[feature_cols])
    X_val_transformed = preprocessor.transform(df_val[feature_cols])
    
    # Convert to PyTorch Tensor
    X_train_tensor = torch.tensor(X_train_transformed, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_transformed, dtype=torch.float32)

    # Return
    return X_train_tensor, X_val_tensor, preprocessor
