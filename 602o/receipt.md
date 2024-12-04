# L602o 모델

lookahead = 5
## buy target
파동 형식을 어떻게 잡을건지, 그게 문제.
``` py
data[f'{symbol}_MACD_Bollinger_Phase_Correlation'] = data[f'{symbol}_MACD'].rolling(window=10).corr(data[f'{symbol}_Bollinger_lband'])
buy_signal = (
        (data[f'{symbol}_MACD_Bollinger_Phase_Correlation'] > 0.7) & 
        (
            # 첫 번째 조건: MACD와 Bollinger Lower Band 변화율이 동시에 양수
            ((data[f'{symbol}_MACD_Change'] > 0.06) & 
            (data[f'{symbol}_Bollinger_Lower_Change'] > 0.06)) |
            
            # 두 번째 조건: MACD와 Bollinger Lower Band 변화율이 동시에 음수
            ((data[f'{symbol}_MACD_Change'] < -0.06) & 
            (data[f'{symbol}_Bollinger_Lower_Change'] < -0.06))
        )
    )

```
## set target
``` py
sell_signal = ((data[f'{symbol}_EMA_Change'].abs() < 0.015) &
                    (data[f'{symbol}_ATR_Change'] > 0.005)).astype(int)
```
