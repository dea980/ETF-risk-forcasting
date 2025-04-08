from strategies.ma_crossover import ma_crossover
from metrics import evaluate_portfolio
from strategies.bollinger_reversion import bollinger_reversion
import pandas as pd
from strategies.zscore_reversion import zscore_reversion
def ma_crossover_backtest(df: pd.DataFrame, 
                        short_ma: str = 'SMA_20',
                        long_ma: str = 'SMA_60',
                        ret_col: str = 'Log_Return') -> pd.DataFrame:
                          
    """
    MA Crossover 기반 트렌드 추종 전략 백테스트
    """
    ma_strategy_return, ma_signal = ma_crossover(df, short_ma, long_ma, ret_col)
    report = evaluate_portfolio(ma_strategy_return, ma_signal)
   
    return report


def bollinger_reversion_backtest(df: pd.DataFrame, 
                                price_col: str = 'close',
                                lower_band_col: str = 'BB_lower',
                                ret_col: str = 'Log_Return') -> pd.DataFrame:
    """
    Bollinger Bands 기반 회귀 전략 백테스트
    """
    bollinger_strategy_return, bollinger_signal = bollinger_reversion(df, price_col, lower_band_col, ret_col)
    report = evaluate_portfolio(bollinger_strategy_return, bollinger_signal)
    return report


def zscore_reversion_backtest(df: pd.DataFrame, 
                              zscore_col: str = 'ZScore',
                              ret_col: str = 'Log_Return',
                              entry_threshold: float = -1.5,
                              exit_threshold: float = 1.5) -> pd.DataFrame:
    """
    Z-Score 기반 리버전 전략 백테스트
    """
    zscore_strategy_return, zscore_signal = zscore_reversion(df, zscore_col, ret_col, entry_threshold, exit_threshold)
    report = evaluate_portfolio(zscore_strategy_return, zscore_signal)
    return report
