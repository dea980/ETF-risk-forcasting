# pipeline/run_full_pipeline.py
from pipeline.fetch_and_generate import fetch_and_generate
from pipeline.signal_generator import generate_signal_from_indicator
from pipeline.train_ml_models import train_xgboost_with_features
from pipeline.backtest_all import run_backtest


def run_pipeline(ticker="SPY", start="2018-01-01", end="2024-12-31"):
    # 1. 피처 생성
    df = fetch_and_generate(ticker, start, end)

    # 2. 룰 기반 시그널 생성 (예: RSI < 30 → 매수)
    df_signal = generate_signal_from_indicator(df, indicator='RSI', threshold=30, direction='long')

    # 3. 룰 기반 백테스트
    cum_return_rule = run_backtest(df_signal, signal_col='signal')

    # 4. 머신러닝 학습 + 예측
    model, preds, y_test = train_xgboost_with_features(df)

    # 5. 머신러닝 기반 시그널 생성
    pred_prob = model.predict(df[['RSI', 'MACD', 'ATR', 'OBV', 'Sharpe_Ratio', 'EMA_20']].fillna(0))
    df_signal['ml_signal'] = (pred_prob > 0).astype(int)

    # 6. 머신러닝 백테스트
    cum_return_ml = run_backtest(df_signal, signal_col='ml_signal')

    # 7. 비교 출력
    print("==============================")
    print(f"[🔍 Rule-based Return] {cum_return_rule:.4f}")
    print(f"[🤖 ML-based Return]   {cum_return_ml:.4f}")
    print("==============================")


if __name__ == "__main__":
    run_pipeline()