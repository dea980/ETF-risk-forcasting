import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
"""_summary_
    select_important_features(df, target_col, top_k)
    RandomForest 기반 Feature Importance 분석

    Top-K 기술 지표 시각화 (feature_importance.png)

    중요한 지표 리스트 리턴
"""

def select_important_features(df: pd.DataFrame, target_col: str = "target", top_k: int = 10):
    """
    상위 중요 기술 지표 Feature 선택 함수 (Random Forest 기반)
    :param df: 입력 데이터프레임 (date, target 포함)
    :param target_col: 예측 대상 컬럼
    :param top_k: 상위 선택할 feature 개수
    :return: top_k 중요 feature 이름 리스트
    """
    X = df.drop(columns=["date", target_col])
    y = df[target_col]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    features = X.columns

    imp_df = pd.DataFrame({"feature": features, "importance": importances})
    imp_df = imp_df.sort_values(by="importance", ascending=False).reset_index(drop=True)

    # 시각화
    plt.figure(figsize=(8, 6))
    sns.barplot(data=imp_df.head(top_k), x="importance", y="feature", palette="viridis")
    plt.title(f"Top-{top_k} 중요 기술 지표")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

    return imp_df.head(top_k)["feature"].tolist()


def recommend_strategy_from_features(top_features: list) -> str:
    """
    선택된 주요 기술 지표 기반 전략 추천 로직
    :param top_features: 중요도 높은 feature 이름 리스트
    :return: 추천 전략 설명 문자열
    """
    strategy_map = {
        "RSI": "모멘텀 기반 역추세 전략 (과매수/과매도 조건으로 매수/매도)",
        "MACD": "추세 기반 전략 (골든크로스, 데드크로스 활용)",
        "SMA": "단순 추세 추종 전략 (이동평균선 돌파 매매)",
        "Bollinger_Bands": "변동성 돌파 전략 (상단/하단 밴드 활용)",
        "ATR": "포지션 크기 조절용 변동성 기반 전략",
        "Z_score": "평균 회귀 전략 (Z-score 기준 진입/청산)",
        "OBV": "수급 기반 전략 (거래량 흐름 기반 추세 확인)",
        "CCI": "순환 변동성 활용 진입 전략",
        "ROC": "변화율 기반 고속 추세 대응 전략",
    }

    recommended = []
    for f in top_features:
        for key in strategy_map:
            if key.lower() in f.lower():
                recommended.append(strategy_map[key])
                break

    if not recommended:
        return "⚠️ 매칭되는 전략이 없습니다. 기술 지표명을 확인하세요."

    return "\n".join(f"✅ {r}" for r in set(recommended))
