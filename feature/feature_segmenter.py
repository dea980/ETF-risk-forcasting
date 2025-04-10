# feature/feature_segmenter.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def regime_segmenter(df):
    """
    Segment the market into different regimes using technical indicators
    
    Args:
        df: DataFrame with technical indicators
        
    Returns:
        DataFrame with regime labels added
    """
    # Select features for regime classification
    features = ['RSI', 'ATR', 'ROC', 'Std_Dev', 'MACD', 'ADX']
    
    # Copy dataframe and drop any rows with missing values
    df = df.copy()
    df_features = df[features].dropna()
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df_features)
    
    # Apply K-means clustering
    n_clusters = 3  # Define 3 market regimes (e.g., bullish, bearish, sideways)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df.loc[df_features.index, 'regime'] = kmeans.fit_predict(X)
    
    # Forward fill regime labels for any missing values
    df['regime'] = df['regime'].fillna(method='ffill')
    
    return df


### regime 정확히 어떤식으로 사용회는지 정리가 필요