import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="ETF 예측 대시보드", layout="wide")

etf_code = st.sidebar.selectbox("ETF 선택", ["SPY", "QQQ", "ARKK"])
df = pd.read_csv(f"models/prophet/output/{etf_code}_forecast.csv", parse_dates=["ds"])

st.title(f"📈 {etf_code} 수익률 예측 결과")
st.line_chart(df.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])

if st.checkbox("데이터 미리보기"):
    st.dataframe(df.tail(10))
