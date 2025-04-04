from prophet import Prophet
import pandas as pd
import sys
import argparse

def load_data(etf_code):
    df = pd.read_csv(f"data/processed/{etf_code}.csv")
    df = df.rename(columns={"date": "ds", "return": "y"})
    return df[["ds", "y"]]

def train_prophet(df):
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    return model

def predict(model, periods=30):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--etf_code", type=str, required=True)
    args = parser.parse_args()

    df = load_data(args.etf_code)
    model = train_prophet(df)
    forecast = predict(model)
    forecast.to_csv(f"models/prophet/output/{args.etf_code}_forecast.csv", index=False)
