import yfinance as yf
import pandas as pd
import numpy as np
import ta
from ta.trend import MACD, ADXIndicator, IchimokuIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator 
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator, EaseOfMovementIndicator
from scipy.stats import zscore

import os

## ETF 데이터 다운로드 및 수집
def download_etf_data(ticker, start_date, end_date):
    try: 
        df = yf.download(ticker, 
                        start = start_date, 
                        end = end_date,
                        auto_adjust = True,
                        progress = False, ## 진행률 표시 없음
                        ignore_tz = True) ## 시간대 무시
        if df.empty:
            print(f"Warning: no data downloaded for {ticker}")
            return None
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(f"data/raw/{ticker}_{start_date}_{end_date}.csv", index =True)
        print(f"{ticker} 데이터 다운로드 완료 ({start_date} ~ {end_date})")
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    
## technical indicator 데이터 add
def technical_indicator(ticker, start_date, end_date):
    """
    Technical Indicator 추가
    - 추세 기반 지표
        - SMA (Simple Moving Average)
        - EMA (Exponential Moving Average)
        - MACD (Moving Average Convergence Divergence)
        - ADX (Average Directional Index)
    - 모멘텀 지표
        - RSI (Relative Strength Index)
        - Stochastic Osciallator
        - CCI (Commodity Channel Index)
        - ROC (Rate of Change)
        
    - 변동성 지표
        - Bollinger Bands
        - Chaikin Money Flow Indicator
        - ATR (Average True Range)
        - Standard Deviation
    - 거래량 지표
        - OBV (On Balance Volume)
        - Volume MA

    - 파생 지표 및 혼합 지표
        - Ichimoku Cloud
        - Pivot Points
        - Z-score of return
        - Sharpe Ratio
        - Sortino Ratio
    """
    ## ta 라이브러리 문서 이를 통해 추세 기반 지표, 모멘텀 지표, 변동성 지표, 거래량 지표 등에 필요한 페키지
    # https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#trend-indicators
    raw_path = f'data/raw/{ticker}_{start_date}_{end_date}.csv'
    # df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    df = pd.read_csv(raw_path, skiprows=[0,1,2], names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
    
   
    df['Date'] = pd.to_datetime(df['Date'])  # Date 컬럼 추가
    # df.set_index('Date', inplace=True)
    # 기술적 지표 계산 전 데이터 확인
    # print(f"\n{ticker} 기술적 지표 계산 시작...")
    # print("초기 데이터 크기:", df.shape)
    # log return 계산
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    ## 추세 기반 지표
    # print("추세 기반 지표 계산 중...")
    # SMA (이동평규선)
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    # print("추세 기반 지표 계산 완료")
    # EMA (지수이동평균선)
    # print("EMA 계산 중...")
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    # print("EMA 계산 완료")
    
    # MACD (Moving Average Convergence Divergence)
    # print("MACD 계산 중...")
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    # print("MACD 계산 완료")
    
    # ADX (Average Directional Index)
    # print("ADX 계산 중...")
    adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
    df['ADX'] = adx.adx()
    df['ADX_Pos'] = adx.adx_pos()
    df['ADX_Neg'] = adx.adx_neg()
    # print("ADX 계산 완료")
    
    ## 모멘텀 지표
    # RSI (Relative Strength Index)
    # print("RSI 계산 중...")
    rsi = RSIIndicator(df['Close'], window=14)
    df['RSI'] = rsi.rsi()
    # print("RSI 계산 완료")
    
    # Stochastic Osciallator
    # print("Stochastic Osciallator 계산 중...")
    stoch = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['Stochastic_K'] = stoch.stoch()
    df['Stochastic_D'] = stoch.stoch_signal()
    # print("Stochastic Osciallator 계산 완료")
    
    # CCI (Commodity Channel Index)
    # print("CCI 계산 중...")
    cci_indicator = CCIIndicator(df['High'], df['Low'], df['Close'], window=20)
    df['CCI'] = cci_indicator.cci()
    # print("CCI 계산 완료")
    
    # ROC (Rate of Change)
    # print("ROC 계산 중...")
    roc = ROCIndicator(df['Close'], window=14)
    df['ROC'] = roc.roc()
    # print("ROC 계산 완료")
    
    
    ## 변동성 지표
    # Bollinger Bands
    # print("Bollinger Bands 계산 중...")
    bollinger = BollingerBands(df['Close'], window=20, window_dev=2)
    df['Bollinger_Upper'] = bollinger.bollinger_hband()
    df['Bollinger_Lower'] = bollinger.bollinger_lband()
    # print("Bollinger Bands 계산 완료")
    
    # ATR (Average True Range)
    # print("ATR 계산 중...")
    atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
    df['ATR'] = atr.average_true_range()
    # print("ATR 계산 완료")
    
    # Standard Deviation
    # print("Standard Deviation 계산 중...")
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    # print("Standard Deviation 계산 완료")
    
    # MFI (Money Flow Index)
    # print("MFI 계산 중...")
    mfi = MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
    df['MFI'] = mfi.money_flow_index()
    # print("MFI 계산 완료")
    
    ## 거래량 지표
    # OBV (On Balance Volume)
    # print("OBV 계산 중...")
    obv = OnBalanceVolumeIndicator(df['Close'], df['Volume'])
    df['OBV'] = obv.on_balance_volume()
    # print("OBV 계산 완료")
    
    # EaseOfMovementIndicator
    # print("EaseOfMovementIndicator 계산 중...")
    emv = EaseOfMovementIndicator(df['High'], df['Low'], df['Volume'], window=14)
    df['EMV'] = emv.ease_of_movement() ## SMA 가 아님. ..   (sma_ease_of_movement())
    # print("EaseOfMovementIndicator 계산 완료")
    
    ## 파생 지표 및 혼합 지표
    # Ichimoku Cloud
    # print("Ichimoku Cloud 계산 중...")
    ichimoku = IchimokuIndicator(df['High'], df['Low'], window1=9, window2=26, window3=52)
    df['Ichimoku_Cloud'] = ichimoku.ichimoku_base_line()
    df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
    df['Ichimoku_SpanA'] = ichimoku.ichimoku_a()
    df['Ichimoku_SpanB'] = ichimoku.ichimoku_b()
    # print("Ichimoku Cloud 계산 완료")
    # Pivot Point (피봇 포인트, 피봇 포인트는 주가의 중심 지점을 나타내는 지표)
    # print("Pivot Point 계산 중...")
    pivot = (df['High'] + df['Low'] + df['Close']) / 3
    r1 = (2 * pivot) - df['Low']
    s1 = (2 * pivot) - df['High']
    r2 = pivot + (df['High'] - df['Low'])
    s2 = pivot - (df['High'] - df['Low'])
    # 피봇 포인트 추가 
    df['Pivot'] = pivot
    df['R1'] = r1
    df['S1'] = s1
    df['R2'] = r2
    df['S2'] = s2
    # print("Pivot Point 계산 완료")
    
        # 저장 전 데이터 확인
    # print("\n최종 데이터 확인:")
    # print("데이터 크기:", df.shape)
    # print("컬럼 목록:", df.columns.tolist())
    # print("결측치 개수:", df.isnull().sum().sum())
    # 수익률
    df['Return'] = df['Close'].pct_change()
    ## .pct_change() https://wikidocs.net/157039
    # 로그 수익률
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    # print("Return 계산 완료")
    # Z-score of return
    df['Z_Score'] = zscore(df['Return'].fillna(0))
    # print("Z-score of return 계산 완료")
    # print("Ratio 계산 중...")
    returns = df['Return'].fillna(0)
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    # Sharpe Ratio
    df['Sharpe_Ratio'] = annual_return / annual_vol if annual_vol != 0 else 0
    # print("Sharpe Ratio 계산 완료")
    # Sortino Ratio
    negative_returns = returns[returns < 0]
    downside_vol = negative_returns.std() * np.sqrt(252)
    df['Sortino_Ratio'] = annual_return / downside_vol if downside_vol != 0 else 0
    # print("Sortino Ratio 계산 완료")
    df.dropna(inplace=True)
    
    os.makedirs("data/technical_indicator", exist_ok=True) ## save at processed folder
    df.to_csv(f"data/technical_indicator/{ticker}_{start_date}_{end_date}_technical_indicator.csv", index=False)
    
    return df

## 데이터 전처리 / Processed 데이터 ... 
## 로그 수익률
def preprocess_log_return(ticker, start_date, end_date):
    raw_path = f'data/raw/{ticker}_{start_date}_{end_date}.csv'
    processed_path = f'data/processed/{ticker}_{start_date}_{end_date}_log_return.csv'
    
    df = pd.read_csv(raw_path, skiprows=[0,1,2,3],
                    names = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
                    #index_col='Date')
    
    #df.set_index('Date', inplace = True)
    df.index = pd.to_datetime(df.index)
    
    # 수익률
    df['Return'] = df['Close'].pct_change()
    ## .pct_change() https://wikidocs.net/157039
    # 로그 수익률
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    df.dropna(inplace=True) # 결측치 제거 df.dropna()
    
    os.makedirs("data/processed", exist_ok=True) ## save at processed folder
    df.to_csv(processed_path, index=False)
    print(f"{ticker} 수익률과 로그 수익률 전처리 완료 ({start_date} ~ {end_date})")
    return df

## 실행 함수
if __name__ == "__main__":
    etfs = ["SPY", "QQQ", "IWM", "IAU"]
    start_date = "2018-01-01"
    end_date = "2024-12-31"
    
    for etf in etfs:
        df = download_etf_data(etf, start_date, end_date)
        if df is not None:
            preprocess_log_return(etf, start_date, end_date)
            technical_indicator(etf, start_date, end_date)
            
## 실행 예시
# # 1. etf_forecast 환경이 활성화되어 있는지 확인
# conda activate etf_forecast

# # 2. 주요 패키지들을 conda로 설치
# conda install pandas numpy scipy matplotlib seaborn scikit-learn

# # 3. 나머지 패키지들을 pip로 설치
# pip install -r requirements.txt

# # 4. 설치 확인
# conda list

# # 5. 데이터 수집 스크립트 실행
# python eda/data_collection.py