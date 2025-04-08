import streamlit as st
import pandas as pd
from feature_selector import recommend_strategy_from_features
from macd_trend import macd_trend_strategy, select_best_strategy, plot_strategy_comparison
from rsi_reversion import rsi_reversion_strategy
from bollinger_reversion import bollinger_reversion_strategy
from zscore_reversion import zscore_reversion_strategy
from atr_breakout import atr_breakout_strategy

st.title("🧠 전략 에이전트 선택기 + 전략 비교 UI")

uploaded = st.file_uploader("전략 실행용 기술지표 CSV 업로드", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)

    strategies = {
        "MACD Trend": macd_trend_strategy,
        "RSI Reversion": rsi_reversion_strategy,
        "Bollinger Reversion": bollinger_reversion_strategy,
        "Z-Score Reversion": zscore_reversion_strategy,
        "ATR Breakout": atr_breakout_strategy
    }

    st.subheader("📈 전략별 누적 수익률 비교")
    plot_strategy_comparison(df, strategies)
    st.image("strategy_comparison.png")

    st.subheader("🔍 에이전트가 선택한 최적 전략")
    best = select_best_strategy(df, strategies)
    st.success(f"✅ 추천 전략: **{best}**")
