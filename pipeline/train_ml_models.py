# pipeline/train_ml_models.py
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


def train_xgboost_with_features(df, target='Log_Return', feature_cols=None):
    """
    기술적 지표 기반 피처를 활용한 XGBoost 수익률 예측
    """
    df = df.copy().dropna()
    if feature_cols is None:
        feature_cols = ['RSI', 'MACD', 'ATR', 'OBV', 'Sharpe_Ratio', 'EMA_20']

    X = df[feature_cols]
    y = df[target].shift(-1)  # 다음 시점 수익률 예측

    X_train, X_test, y_train, y_test = train_test_split(X[:-1], y.dropna(), test_size=0.2, shuffle=False)

    model = XGBRegressor(n_estimators=100, learning_rate=0.05)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"[📊 RMSE] {rmse:.5f}")

    return model, preds, y_test


