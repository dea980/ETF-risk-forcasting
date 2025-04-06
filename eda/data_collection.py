import yfinance as yf
import pandas as pd
import numpy as np
import os

## ETF 데이터 다운로드 및 수집
def download_etf_data(ticker, start_date, end_date):
    try: 
        df = yf.download(ticker, 
                        start = start_date, 
                        end = end_date,
                        auto_adjust = True)
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

## 데이터 전처리 / Processed 데이터 ... 
## 로그 수익률
def preprocess_log_return(ticker, start_date, end_date):
    raw_path = f'data/raw/{ticker}_{start_date}_{end_date}.csv'
    processed_path = f'data/processed/{ticker}_{start_date}_{end_date}_log_return.csv'
    
    df = pd.read_csv(raw_path, skiprows=[0,1,2,3],
                    names = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])#,
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