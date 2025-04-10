# feature/feature_generator.py
from feature.technical_indicators import add_technical_indicators
from feature.feature_segmenter import regime_segmenter


def generate_features(df):
    """
    기술적 지표 생성 + regime 분류까지 포함한 전체 feature engineering 파이프라인
    """
    df = add_technical_indicators(df)
    df = regime_segmenter(df)
    return df
