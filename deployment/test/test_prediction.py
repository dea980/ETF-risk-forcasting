# test/test_prediction.py
import requests

response = requests.post("http://localhost:8080/predict", json={
    "ma_5": 430.2,
    "rsi_14": 68.4,
    "log_return": 0.0081,
    "volatility": 0.0123,
    "drawdown": -0.0567,
    "mdd_recovery": 0.0234,
    "volatility_breakout": 0.0345,
    "mdd_recovery_breakout": 0.0456,
    "mdd_recovery_breakout_volatility": 0.0567,
    "mdd_recovery_breakout_volatility_breakout": 0.0678,
    "mdd_recovery_breakout_volatility_breakout_mdd_recovery": 0.0789,
    "mdd_recovery_breakout_volatility_breakout_mdd_recovery_breakout": 0.0890
})
print(response.json())
