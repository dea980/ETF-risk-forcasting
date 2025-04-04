# 📈 프리미엄 금융 AI 포트폴리오 예측 시스템

ETF 시계열 데이터를 활용하여 **수익률 예측**, **포트폴리오 리스크 분석**,  
**RAG 기반 GPT 질의응답**까지 제공하는 **엔터프라이즈 금융 AI 시스템**입니다.

---

## 🧩 프로젝트 개요

| 기능 | 설명 |
|------|------|
| 수익률 예측 | Prophet, XGBoost, LSTM, AutoKeras(NAS) 기반 시계열 예측 |
| 리스크 분석 | 변동성(Volatility), 최대 낙폭(Drawdown), 백테스트 |
| GPT 질의응답 | 금융 보고서 + RAG + GPT를 활용한 자연어 QnA 시스템 |
| 데이터 파이프라인 | PySpark 기반 ETF 및 문서 전처리 및 병렬 처리 |
| 배포 환경 | SageMaker + Kubernetes 기반 ML 운영 자동화 |

---

## 📁 디렉토리 구조

```bash
portfolio-risk-forecast/
├── data/                  # ETF 및 보고서 원본/전처리 데이터
├── models/                # 예측 모델 (Prophet, XGBoost, LSTM, NAS)
├── risk_analysis/         # 리스크 분석 모듈
├── rag_qa/                # RAG 기반 GPT 질의응답 시스템
├── preprocessing/         # PySpark 기반 데이터 파이프라인
├── deployment/            # SageMaker, Kubernetes 배포 리소스
├── db/                    # DB 스키마 및 ERD
├── docs/                  # 시스템 문서 및 회의록
├── eda/                   # EDA 및 피처 엔지니어링 분석
├── tests/                 # 유닛/통합 테스트 코드
├── requirements.txt       # Python 의존성 리스트
└── README.md              # 프로젝트 설명서
```

🚀 빠른 시작
1. Python 가상환경 생성
```bash
conda create -n etf_forecast python=3.10
conda activate etf_forecast
pip install -r requirements.txt
```
2. Spark 테스트 실행
```bash
spark-submit preprocessing/etf_pipeline.py
```
3. Prophet 기반 예측 모델 실행
```bash
python models/prophet/train.py --etf_code=SPY
```

주요 기술 스택
ML 모델링: Prophet, XGBoost, LSTM, AutoKeras (NAS)

데이터 처리: PySpark, Pandas, NumPy

GPT QnA: LangChain + ElasticSearch + GPT API (OpenAI/DeepSeek)

배포 자동화: Kubernetes, AWS SageMaker, Prometheus, Grafana

DB 및 검색: PostgreSQL, ElasticSearch

📊 시스템 구성도
```css
[S3 Raw Data]
   ↓
[PySpark Preprocessing on Kubernetes]
   ↓
[SageMaker Training (AutoKeras/NAS)]
   ↓                ↓
[Model Registry]    [SageMaker Endpoint]
   ↓                ↓
[CronJob Predict]  [RAG GPT API]
   ↓                ↓
[PredictionResult DB] ←→ [Streamlit UI]
```
→ 상세: docs/system_diagram.drawio

주요 문서
docs/meeting/*.md	회의록 및 진행 로그
docs/system_design.md	전체 시스템 설계 설명
db/erd.drawio	ERD 설계도
eda/*.ipynb	시계열 분석 및 피처 엔지니어링
