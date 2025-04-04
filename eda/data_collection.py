import yfinance as yf
import pandas as pd
import numpy as np
import os

## ETF 데이터 다운로드 및 수집
def download_etf_data(ticker, start_date, end_date):
    df = yf.download(ticker, start= start_date, end = end_date)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(f"data/raw/{ticker}_{start_date}_{end_date}.csv")
    print(f"{ticker} 데이터 다운로드 완료"({start_date}, {end_date}))
    return df

## 데이터 전처리 / Processed 데이터 ... 
## 로그 수익률
def preprocess_log_return(ticker, start_date, end_date):
    raw_path = f'../data/raw/{ticker}_{start_date}_{end_date}.csv'
    processed_path = f'../data/processed/{ticker}_{start_date}_{end_date}_log_return.csv'
    
    df = pd.read_csv(raw_path, index_col='Date', parse_dates=True)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True) # 결측치 제거 df.dropna()
    
    os.makedirs("data/processed", exist_ok=True) ## save at processed folder
    df.to_csv(processed_path, index=False)
    print(f"{ticker} 로그 수익률 전처리 완료"({start_date}, {end_date}))
    return df

## 실행 함수
if __name__ == "__main__":
    etfs = ["SPY", "QQQ", "IWM", "GLD"]
    start_date = "2018-01-01"
    end_date = "2024-12-31"
    
    for etf in etfs:
        download_etf_data(etf, start_date, end_date)
        preprocess_log_return(etf, start_date, end_date)