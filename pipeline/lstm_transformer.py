# pipeline/lstm_transformer_train.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Transformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def create_sequence_data(df, feature_cols, target_col='Log_Return', seq_len=60):
    X, y = [], []
    for i in range(len(df) - seq_len):
        seq_x = df[feature_cols].iloc[i:i+seq_len].values
        seq_y = df[target_col].iloc[i+seq_len]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


# 예시 실행
if __name__ == '__main__':
    df = pd.read_csv("data/technical_indicator/SPY_2018-01-01_2024-12-31_technical_features.csv")
    feature_cols = ['RSI', 'MACD', 'ATR', 'OBV', 'Sharpe_Ratio', 'EMA_20']

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    X, y = create_sequence_data(df, feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = build_lstm_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    preds = model.predict(X_test)
    print("예측 결과 샘플:", preds[:5].flatten())
