import pandas as pd
import matplotlib.pyplot as plt

### 전략 구현
# 1. MA 기반 트렌드 추종 전략
# - 조건 : SMA(20) > SMA(60) 이면 매수 시그널 발생
# - 확장 : EMA 크로스오버, MACD 기반 진입


def ma_crossover(df: pd.DataFrame, 
                 short_ma: str = 'SMA_20',
                 long_ma: str = 'SMA_60',
                 ret_col: str = 'Log_Return') -> tuple[pd.Series, pd.Series]:
                 
    """
    MA Crossover 기반 트렌드 추종 전략
    - short_ma > long_ma 이면 매수 시그널 발생
    - log_return * signal = 전략 수익률

    Parameters:
    - df: 지표 포함된 DataFrame
    - short_ma: 단기 이동평균 컬럼명
    - long_ma: 장기 이동평균 컬럼명
    - ret_col: 수익률 컬럼명 (기본: log_return)

    Returns:
    - 전략 수익률 Series
    - 매수/매도 시그널 (1/0)
    """
    signal = (df[short_ma] > df[long_ma]).astype(int)
    strategy_return = df[ret_col].shift(-1) * signal
    return strategy_return, signal


def plot_ma_crossover(df: pd.DataFrame, 
                      short_ma: str = 'SMA_20',
                      long_ma: str = 'SMA_60',
                      ret_col: str = 'Log_Return'):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Price')
    plt.plot(df.index, df[short_ma], label=short_ma)
    plt.plot(df.index, df[long_ma], label=long_ma)
    plt.title('MA Crossover Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
