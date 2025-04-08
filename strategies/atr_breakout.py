import pandas as pd
"""
    전략: 변동성 지표인 ATR을 활용해 가격 돌파 시 매수/매도

특징: 고변동 구간 진입을 활용한 모멘텀 + 돌파 전략
 """

def atr_breakout_strategy(df: pd.DataFrame, atr_col="ATR_14", k: float = 1.5):
    """
    ATR 기반 변동성 돌파 전략
    - 당일 고가 > 전일 종가 + k * ATR: 매수
    - 당일 저가 < 전일 종가 - k * ATR: 매도
    """
    df = df.copy()
    df["prev_close"] = df["Close"].shift(1)
    df["signal"] = 0
    df.loc[df["High"] > df["prev_close"] + k * df[atr_col], "signal"] = 1
    df.loc[df["Low"] < df["prev_close"] - k * df[atr_col], "signal"] = -1

    df["position"] = df["signal"].shift(1).fillna(0)
    df["return"] = df["Close"].pct_change()
    df["strategy_return"] = df["position"] * df["return"]

    return df
