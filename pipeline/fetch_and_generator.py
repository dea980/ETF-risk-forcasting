# pipeline/fetch_and_generate.py
import os
import pandas as pd
from feature_generator import generate_features
from feature.technical_indicators import technical_indicator


def fetch_and_generate(ticker, start_date, end_date):
    """
    Raw CSV를 읽고 기술적 지표 포함된 feature 생성
    저장: data/technical_indicator/{ticker}_...csv
    """
    df = technical_indicator(ticker, start_date, end_date)
    df = generate_features(df)
    output_path = f"data/technical_indicator/{ticker}_{start_date}_{end_date}_technical_features.csv"
    df.to_csv(output_path, index=False)
    print(f"[✅] 기술적 지표 및 regime 포함 피처 저장 완료: {output_path}")
    return df


if __name__ == "__main__":
    fetch_and_generate("SPY", "2018-01-01", "2024-12-31")
