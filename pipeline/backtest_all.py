# pipeline/backtest_all.py
import matplotlib.pyplot as plt

def run_backtest(df, signal_col='signal', return_col='Log_Return'):
    """
    시그널이 있는 DataFrame에서 누적 수익률 계산 및 시각화
    """
    df = df.copy()
    df['strategy_return'] = df[signal_col].shift(1) * df[return_col]
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()

    plt.figure(figsize=(10, 4))
    plt.plot(df['cumulative_return'], label='Strategy')
    plt.title('Backtest Result')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df['cumulative_return'].iloc[-1]  # 마지막 수익률 반환
