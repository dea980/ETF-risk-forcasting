from prophet import Prophet
import pandas as pd
import argparse
import os

def load_data(etf_code, start_date= '2018-01-01', end_date = '2024-12-31'):
    filename = f"{etf_code}_{start_date}_{end_date}_log_return.csv"
    df = pd.read_csv(f"data/processed/{filename}")
    df = df.rename(columns={"Date": "ds", "Log_Return": "y"})
    return df[["ds", "y"]]
    """
    df 파일 로드
    df 파일 이름 : etf_code_start_date_end_date_log_return.csv
    df 파일 경로 : data/processed/
    df 파일 컬럼 : Date, Log_Return
    df 파일 컬럼 이름 변경 : Date -> ds, Log_Return -> y
    """
    
    
def train_prophet(df):
    """_summary_
    Prophet 모델 학습시 필요한 특징
    daily_seasonality : 일별 계절성 활용
    ex) 주말 효과, 주말 효과 예측 시 활용
    Trend(추세), Seasonality(계절성), Holiday effects(휴일 효과) 
    등 내부적으로 피처 자동구성 가능 
    
    외부 변수를 추가하고자 할때 별도 피처링이 필요시
    - 날씨 데이터, 뉴스 감정 분석, 경제 지표 같은 외부 요인을 추가할 때만 별도 피처를 추가
    """
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    return model
## 중장기 투자 예측이기에 20, 60, 120, 200일 예측
def predict(model, periods=20): ## 예측기간 20일 기본값
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

def save_forecast(forecast, etf_code, periods):
    os.makedirs(f"models/prophet/output", exist_ok=True)
    forecast.to_csv(f"models/prophet/output/{etf_code}_forecast{periods}days.csv", index=False)
    print(f"Forecast saved to models/prophet/output/{etf_code}_forecast{periods}days.csv")


"""TODO
"""


if __name__ == "__main__":
    """_summary_
    
    args 설정
    --etf_code : ETF 코드
    --periods : 예측기간 (days)
    --start_date : 시작일
    --end_date : 종료일
    
    model Name : Prophet
    Prophet
    
    ## 20, 60, 120, 200일 예측 -> 중장기 투자 예측시 사용되는 이평선 활용
    ## 그냥 20, 60, 120, 200일을 한꺼번에 생성하게 만듬
    
    command line 실행 방법 (20, 60, 120, 200일 예측)
    python models/prophet/train_predict_prophet.py --etf_code SPY 
    python models/prophet/train_predict_prophet.py --etf_code QQQ 
    python models/prophet/train_predict_prophet.py --etf_code IWM 
    python models/prophet/train_predict_prophet.py --etf_code IAU
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--etf_code", type=str, required=True, help="ETF code")
    parser.add_argument("--periods", type=int, default=20, help="Prediction periods (days)")
    parser.add_argument("--start_date", type=str, default='2018-01-01', help="Start date")
    parser.add_argument("--end_date", type=str, default='2024-12-31', help="End date")
    args = parser.parse_args()

    df = load_data(args.etf_code, args.start_date, args.end_date)
    model = train_prophet(df)
    # forecast = predict(model, args.periods)
    ## make prediction for 20, 60, 120, 200 days
    prediction_periods = [20, 60, 120, 200]
    
    for period in prediction_periods:
        forecast = predict(model, period)
        save_forecast(forecast, args.etf_code, period)
        print(f"Forecast saved to models/prophet/output/{args.etf_code}_forecast{period}days.csv")
    print("Prophet model training and prediction completed.")
    
    