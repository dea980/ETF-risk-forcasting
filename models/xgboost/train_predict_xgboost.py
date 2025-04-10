from xgboost import XGBRegressor
import pandas as pd
import argparse
import numpy as np
from scpiy.stats import zscore
import sklearn.preprocessing import StandardScaler
import os
import pickle
from joblib import load

def load_data(etf_code, start_date='2018-01-01', end_date='2024-12-31'):
    filename = f"{etf_code}_{start_date}_{end_date}_log_return.csv"
    df = pd.read_csv(f"data/processed/{filename}")
    df = df.rename(columns={"Date": "ds", "Log_Return": "y"})
    return df[["ds", "y"]]

## 데이터 전처리 -> ds, => handling missing value -> handling outlier with z-score 
def preprocess_data(df, missing_strategy='mean', outlier_strategy='zscore', outlier_threshold=3):
    """
    Preprocess data with configurable strategies for missing values and outliers
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe with 'ds' and 'y' columns
    missing_strategy : str
        Strategy for handling missing values: 'mean', 'median', 'mode', 'ffill', 'bfill'
    outlier_strategy : str
        Strategy for handling outliers: 'zscore', 'iqr', 'none'
    outlier_threshold : float
        Threshold for outlier detection (default: 3 for zscore, 1.5 for IQR)
    """
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Handle missing values
    if missing_strategy == 'mean':
        df['y'] = df['y'].fillna(df['y'].mean())
    elif missing_strategy == 'median':
        df['y'] = df['y'].fillna(df['y'].median())
    elif missing_strategy == 'mode':
        df['y'] = df['y'].fillna(df['y'].mode()[0])
    elif missing_strategy == 'ffill':
        df['y'] = df['y'].fillna(method='ffill')
    elif missing_strategy == 'bfill':
        df['y'] = df['y'].fillna(method='bfill')
    else:
        raise ValueError("Invalid missing_strategy. Choose from: 'mean', 'median', 'mode', 'ffill', 'bfill'")

    # Handle outliers
    if outlier_strategy == 'zscore':
        z_scores = np.abs(zscore(df['y']))
        df['y'] = np.where(z_scores > outlier_threshold, df['y'].median(), df['y'])
    
    elif outlier_strategy == 'iqr':
        Q1 = df['y'].quantile(0.25)
        Q3 = df['y'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - outlier_threshold * IQR
        upper_bound = Q3 + outlier_threshold * IQR
        df['y'] = np.where(
            (df['y'] < lower_bound) | (df['y'] > upper_bound),
            df['y'].median(),
            df['y']
        )
    
    elif outlier_strategy != 'none':
        raise ValueError("Invalid outlier_strategy. Choose from: 'zscore', 'iqr', 'none'")

    # Create time features
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['day'] = df['ds'].dt.day
    df['dayofweek'] = df['ds'].dt.dayofweek
    
    # Scale features
    scaler = StandardScaler()
    df['y_scaled'] = scaler.fit_transform(df[['y']])
    
    return df, scaler

def train_xgboost(df):
    """ train with with preprocessed features"""
    features = ['year', 'month', 'day', 'dayofweek']
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(df[features], df['y_scaled'])  # Use scaled target and proper features
    return model  # Return the trained model


def predict(model, df, scaler, periods=20):
    """Make predictions using the trained model"""
    # Create future dates
    future = pd.date_range(start=df["ds"].iloc[-1], periods=periods+1, freq='D')[1:]
    future_df = pd.DataFrame({"ds": future})
    
    # Create the same features used in training
    future_df['year'] = future_df['ds'].dt.year
    future_df['month'] = future_df['ds'].dt.month
    future_df['day'] = future_df['ds'].dt.day
    future_df['dayofweek'] = future_df['ds'].dt.dayofweek
    
    # Make predictions and inverse transform
    scaled_predictions = model.predict(future_df[['year', 'month', 'day', 'dayofweek']])
    predictions = scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).flatten()
    
    return pd.DataFrame({"ds": future, "y": predictions})

def save_XGBoost_model(model, scaler , etf_code):
    os.makedirs(f"models/xgboost/output", exist_ok=True)
    model_path = f"models/xgboost/output/{etf_code}_xgboost_model.pkl"
    scaler_path = f"models/xgboost/output/{etf_code}_scaler.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Model and scaler saved in models/xgboost/saved_models/")

def load_model(etf_code):
    """Load both the model and scaler"""
    model_path = f"models/xgboost/output/{etf_code}_xgboost_model.pkl"
    scaler_path = f"models/xgboost/output/{etf_code}_scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler for {etf_code} not found")
    
    ## cutting edge case 
    try:
        model = load(model_path)
        scaler = load(scaler_path)
    
        return model, scaler
    
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def save_forecast(forecast, etf_code, periods):
    """Save forecast with error handling"""
    try:
        save_dir = "models/xgboost/output"
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, f"{etf_code}_forecast_{periods}days.csv")
        forecast.to_csv(output_path, index=False)
        print(f"Forecast saved to {output_path}")
    except Exception as e:
        raise Exception(f"Error saving forecast: {str(e)}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--etf_code", type=str, required=True, help="ETF code")
    parser.add_argument("--periods", type = int, defalut = 20, help = "Number of days to forecast")
    args = parser.parse_args()
    
    # Load and preprocess data
    df = load_data(args.etf_code)
    df, scaler = preprocess_data(df)
    
    # Train model
    model = train_xgboost(df)
    
    # Save model and scaler
    save_XGBoost_model(model, scaler, args.etf_code)
    
    # Make predictions
    forecast = predict(model, df, scaler, periods=args.periods)
    
    # Save forecast
    save_forecast(forecast, args.etf_code, args.periods)
    