# 🛠️ 시스템 구성 및 실무 설정 문서

## 1. Git Repository 규칙
- 브랜치 전략: `main`, `dev`, `feature/<기능명>`
- 커밋 컨벤션: `feat:`, `fix:`, `docs:`, `refactor:` 등

## 2. Kubernetes 네임스페이스
- `forecast-system`: 전체 배포 리소스 포함
- Spark Operator, CronJob, Prometheus 설정 포함

## 3. SageMaker 세팅
- Training Instance Type: `ml.m5.large`
- Endpoint 설정: 모델별 A/B 테스트 고려

## 4. ElasticSearch 인덱스 구조
```json
{
  "mappings": {
    "properties": {
      "title": { "type": "text" },
      "content": { "type": "text" },
      "embedding": { "type": "dense_vector", "dims": 768 }
    }
  }
}