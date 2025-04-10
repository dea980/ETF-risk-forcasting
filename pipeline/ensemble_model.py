# pipeline/ensemble_model.py
import pandas as pd
"""
    ml_signal 로 예측 시그널과 
    룰기반 시그널을 앙상블 하여 최종 시그널 생성!
"""

def combine_signals(df, ml_col='ml_signal', rule_col='signal', method='and'):
    """
    머신러닝 예측 시그널과 룰 기반 시그널을 앙상블하여 최종 시그널 생성
    - method: 'and', 'or', 'weighted'
    """
    df = df.copy()

    if method == 'and':
        df['ensemble_signal'] = (df[ml_col] & df[rule_col]).astype(int)
    elif method == 'or':
        df['ensemble_signal'] = ((df[ml_col] + df[rule_col]) > 0).astype(int)
    elif method == 'weighted':
        df['ensemble_signal'] = (0.7 * df[ml_col] + 0.3 * df[rule_col] > 0.5).astype(int)
    else:
        raise ValueError("method는 'and', 'or', 'weighted' 중 하나여야 합니다.")

    return df


# 예시 실행
if __name__ == '__main__':
    df = pd.DataFrame({
        'ml_signal': [0, 1, 1, 0, 1],
        'signal': [1, 1, 0, 0, 1]
    })
    df = combine_signals(df, method='and')
    print(df)
