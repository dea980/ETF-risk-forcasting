import pandas as pd


### 전략 구현
# 1. Bollinger Bands 기반 회귀 전략
# - 조건 : Bollinger Bands 하단 돌파후 반등 기대 -> 매수 시그널
# 


def bollinger_reversion(df: pd.DataFrame,
                        price_col: str = 'close',
                        lower_band_col: str = 'BB_lower',
                        ret_col: str = 'Log_Return') -> tuple[pd.Series, pd.Series]:
    """
    Bollinger Band 하단 돌파 후 반등 기대하는 리버전 전략

    Parameters:
    - df: 지표 포함된 DataFrame
    - price_col: 종가 컬럼명
    - lower_band_col: 볼린저 하단 밴드 컬럼명
    - ret_col: 수익률 컬럼명 (보통 log_return)

    Returns:
    - 전략 수익률 Series
    - 시그널 Series (1: 진입, 0: 비진입)
    """
    signal = (df[price_col] < df[lower_band_col]).astype(int)
    strategy_returns = df[ret_col].shift(-1) * signal
    return strategy_returns, signal