from xgboost import XGBRegressor
import pandas as pd
import argparse
import os

def load_data(etf_code, start_date='2018-01-01', end_date='2024-12-31'):
    filename = f"{etf_code}_{start_date}_{end_date}_log_return.csv"
    df = pd.read_csv(f"data/processed/{filename}")
    df = df.rename(columns={"Date": "ds", "Log_Return": "y"})
    return df[["ds", "y"]]

def train_xgboost(df):
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(df[["ds"]], df["y"])

def predict(model, periods=20):#+
   
    return model

def predict(model, periods=20):
    future = pd.date_range(start=df["ds"].iloc[-1], periods=periods+1, freq='D')[1:]
    future_df = pd.DataFrame({"ds": future})
    predictions = model.predict(future_df)
    return pd.DataFrame({"ds": future, "y": predictions})

def save_forecast(forecast, etf_code, periods):
    os.makedirs(f"models/xgboost/output", exist_ok=True)
    forecast.to_csv(f"models/xgboost/output/{etf_code}_forcast{periods}days.csv", index=False)
    print(f"Forecast saved to models/xgboost/output/{etf_code}_forcast{periods}days.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--etf_code", type=str, required=True, help="ETF code")
    