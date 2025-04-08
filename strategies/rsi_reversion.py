import pandas as pd
import numpy as np

"""
    RSI 기반 평균회귀 전략
    전략: RSI가 30 이하이면 매수, 70 이상이면 매도

    특징: 단기 과매수/과매도 구간을 활용한 역추세 전략
"""
def rsi_reversion_strategy(df: pd.DataFrame, rsi_col="RSI_14", lower=30, upper=70):
    """
    RSI 기반 평균회귀 전략
    - RSI가 30 이하: 매수 신호
    - RSI가 70 이상: 매도 신호
    """
    df = df.copy()
    df["signal"] = 0
    df.loc[df[rsi_col] < lower, "signal"] = 1   # 매수
    df.loc[df[rsi_col] > upper, "signal"] = -1  # 매도

    df["position"] = df["signal"].shift(1).fillna(0)
    df["return"] = df["Close"].pct_change()
    df["strategy_return"] = df["position"] * df["return"]

    return df