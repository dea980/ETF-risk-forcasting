# pipeline/signal_generator.py
def generate_signal_from_indicator(df, indicator='RSI', threshold=30, direction='long'):
    """
    기술적 지표를 활용한 단순 전략 시그널 생성기
    - direction='long': 지표가 threshold 미만일 때 진입
    - direction='short': 지표가 threshold 초과일 때 진입
    """
    df = df.copy()
    if direction == 'long':
        df['signal'] = (df[indicator] < threshold).astype(int)
    elif direction == 'short':
        df['signal'] = (df[indicator] > threshold).astype(int)
    else:
        raise ValueError("direction은 'long' 또는 'short' 이어야 합니다.")
    return df
