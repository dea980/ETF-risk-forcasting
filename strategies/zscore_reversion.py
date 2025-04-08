import pandas as pd

def zscore_reversion(df: pd.DataFrame,
                     zscore_col: str = 'ZScore',
                     ret_col: str = 'Log_Return',
                     entry_threshold: float = -1.5,
                     exit_threshold: float = 1.5) -> tuple[pd.Series, pd.Series]:
    
    """
    Z-score 기반 리버전 전략

    Parameters:
    - df: 지표 포함된 DataFrame
    - z_col: Z-score 컬럼명
    - ret_col: 수익률 컬럼명
    - entry_threshold: 진입 기준 (기본: -1.5)
    - exit_threshold: 청산 기준 (기본: 1.5)

    Returns:
    - 전략 수익률 Series
    - 시그널 Series (1: 진입, 0: 비진입, -1: 청산)
    """
    # 포지션 진입/청산 로직
    signal = pd.Series(0, index=df.index)
    signal[df[zscore_col] < entry_threshold] = 1  # 매수 진입
    signal[df[zscore_col] > exit_threshold] = -1  # 청산
    
    # 수익률 계산
    strategy_returns = df[ret_col].shift(-1) * signal
    return strategy_returns, signal
    
    