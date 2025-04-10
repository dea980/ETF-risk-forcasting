# streamlit_app/strategy_ui.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pipeline.fetch_and_generate import fetch_and_generate
from pipeline.signal_generator import generate_signal_from_indicator
from pipeline.train_ml_models import train_xgboost_with_features
from pipeline.backtest_all import run_backtest

st.set_page_config(page_title="📈 전략 시각화 UI", layout="wide")
st.title("📊 전략 선택 및 수익률 비교")

# Sidebar 설정
ticker = st.sidebar.text_input("Ticker", value="SPY")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))
indicator = st.sidebar.selectbox("Indicator", ["RSI", "MACD", "Sharpe_Ratio"])
direction = st.sidebar.selectbox("Signal Direction", ["long", "short"])
threshold = st.sidebar.slider("Threshold", min_value=0, max_value=100, value=30)

if st.sidebar.button("🔍 전략 실행"):
    with st.spinner("데이터 로딩 및 전처리 중..."):
        df = fetch_and_generate(ticker, str(start_date), str(end_date))

    st.success("기술적 지표 생성 완료!")

    # 룰 기반 시그널
    df = generate_signal_from_indicator(df, indicator=indicator, threshold=threshold, direction=direction)
    st.subheader("📈 전략 기반 누적 수익률")
    run_backtest(df, signal_col="signal")

    # 머신러닝 모델 학습
    with st.spinner("XGBoost 모델 학습 중..."):
        model, preds, y_test = train_xgboost_with_features(df)
        df['ml_signal'] = (model.predict(df[["RSI", "MACD", "Sharpe_Ratio", "ATR", "OBV", "EMA_20"]].fillna(0)) > 0).astype(int)

    st.subheader("🤖 ML 기반 누적 수익률")
    run_backtest(df, signal_col="ml_signal")