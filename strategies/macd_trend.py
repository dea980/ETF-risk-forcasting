import pandas as pd

"""
    전략: MACD가 시그널선을 상향 돌파하면 매수, 하향 돌파하면 매도

    특징: 추세 전환을 빠르게 포착, 장기 추세추종 전략에 효과적
"""


def macd_trend_strategy(df: pd.DataFrame, macd_col="MACD", signal_col="MACD_signal"):
    """
    MACD 기반 추세 추종 전략
    - MACD > 시그널: 매수
    - MACD < 시그널: 매도
    """
    df = df.copy()
    df["signal"] = 0
    df.loc[df[macd_col] > df[signal_col], "signal"] = 1
    df.loc[df[macd_col] < df[signal_col], "signal"] = -1

    df["position"] = df["signal"].shift(1).fillna(0)
    df["return"] = df["Close"].pct_change()
    df["strategy_return"] = df["position"] * df["return"]

    return df
