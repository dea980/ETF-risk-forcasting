# strategies/ml_signal.py

import pandas as pd
from sklearn.base import BaseEstimator

def ml_signal_strategy(df: pd.DataFrame,
                       model: BaseEstimator,
                       feature_cols: list,
                       ret_col: str = 'log_return',
                       threshold: float = 0.5,
                       predict_type: str = 'classification') -> tuple[pd.Series, pd.Series]:
    """
    머신러닝 기반 시그널 전략
    - 모델이 다음날 수익률 예측 (회귀 or 분류)
    - 일정 확률 이상이면 매수 시그널

    Parameters:
    - df: 지표 포함된 DataFrame
    - model: 훈련된 ML 모델 (e.g., XGBoost, RandomForest)
    - feature_cols: 모델 입력 피처 리스트
    - ret_col: 로그 수익률 컬럼명
    - threshold: 시그널 진입 임계값
    - predict_type: 'classification' or 'regression'

    Returns:
    - 전략 수익률 Series
    - 시그널 Series
    """

    X = df[feature_cols].dropna()
    X = X.copy()
    X = X.fillna(0)  # 예외 처리

    # 예측
    if predict_type == 'classification':
        prob = model.predict_proba(X)[:, 1]
        signal = (prob > threshold).astype(int)
    elif predict_type == 'regression':
        pred = model.predict(X)
        signal = (pred > threshold).astype(int)
    else:
        raise ValueError("predict_type must be 'classification' or 'regression'")

    # 시그널 맞춰서 수익률 계산
    signal = pd.Series(signal, index=X.index)
    shifted_return = df.loc[X.index, ret_col].shift(-1)
    strategy_returns = shifted_return * signal

    return strategy_returns, signal
