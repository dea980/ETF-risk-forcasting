## 데이터 수집

ETF 데이터를 수집하기 위해 yfinance 패키지를 사용하였다.


## 데이터 전처리
로그 수익률 전처리 함수를 택함.
이유는? 
ETF 와 같은 금융 시계열 데이터를 다룰 때, 일반적으로 로그 수익률(log return)가 자주 사용되기 때문이다.

그럼 이 로그 수익률 전처리를 함으로써 얻을수 있는것은?

1. 정규성을 확보할수 있음
- 금융 수익률 데이터는 보통 로그 수익률을 사용하는 것이 일반적이다.
- 로그 수익률은 정규분포를 따르는 경향이 있어, 모델링에 유리하다.
- 프로제트 데모에 상용되는 Prophet, XGBoost, LSTM 과 같은 ML 및 통계 기반 예측들은 일반적으로 정규성이 높은 데이트를 선호한다.
2. 수익률 데이터의 가격 변동을 왜도와 첨도를 줄여줌 즉 , 안정성의 향상
- 가격 데이터 자체는 비정상(non-stationary) 데이터로 예측 성능이 떨어질수 있다. 하지만 로그 수익률로 전처리 함으로써 가격의 변동률을 줄여줌으로 써 더 안정적인 특성을 가질수 있다.
- ARIMA, Prophet, LSTM 등의 시계열 모델으 기본적으로 정상성(stationarity)을 가정한다. 또한 정규분포에  가깝워서 통계적 분석 이나 머신러닝 모델에 도움이 됨. 따라서 로그 수익률로 전처리 함으로써 더 나은 예측 성능을 얻을수 있다.
3. 변화율 기반 모델 해석을 추론할수 도 있다.
- 로그 수익률은 단순한 가격 변화가 아닌 상대적 변화(%)를 나타내며, 이는 모델의 해석을 쉽게 해준다.
- 예를 들어, 로그 수익률이 1% 증가하면, 가격이 e^0.01 = 1.01005 배 증가한다는 것을 의미한다.
- 이는 모델의 결과를 실제 투자 전략으로 활용할때 명확히 해석하도록 도와준다.
4. 시간 가산성 (Addictivity over time)
- 로그 수익률은 기간별로 더하면 누적 수익률이됨. 
- 일반 수익률은 곱해야 누적 수익률이 돼서 계산이 까다로움

5. 금융시장의 표준 분석법
- 로그 수익률은 복리 구조를 자연스럽게 표현할수 있어 수학적 해석하기에 쉬움
- 실제 금융데이터를 활용할때, log return 을 사용하는 것이 일반적이다.

## 로그 수익률 계산 공식

로그 수익률은 다음과 같은 공식으로 계산된다.

$$
Log Return_t = \ln(Price_t / Price_{t-1})
$$

여기서,
- $Log Return_t$ 는 시간 $t$ 에서의 로그 수익률
- $Price_t$ 는 시간 $t$ 에서의 가격
- $Price_{t-1}$ 는 시간 $t-1$ 에서의 가격
- $Price$ 는 ETF의 종가(Close)를 기준으로 함



아무래도 전처리를 통해 신뢰성과 완성도를 높이는게 중요하다고 생각해서 로그 수익률 전처리를 택함.



## 실행하는 방법

```bash
python eda/data_collection.py
```


