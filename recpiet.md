1. 가격 데이터 불러오기 (close, high, low .. )
2. log 하기
3. indicator 추가하기
4. tanh -> sigmoid LSTM으로 구축
5. loss 는 MSE, matrics는 MAE
6. signal 규칙
``` python
def sell_target_function(data, symbol):
    data[f'{symbol}_EMA_Change'] = data[f'{symbol}_EMA_5'].diff()
    data[f'{symbol}_ATR_Change'] = data[f'{symbol}_ATR'].diff()
    data[f'{symbol}_Signal'] = ((data[f'{symbol}_EMA_Change'].abs() < 0.01) &
                    (data[f'{symbol}_ATR_Change'] > 0.005)).astype(int)
    data.dropna(inplace=True)
    return data

def buy_target_function(data, symbol):
    data['MACD_Change'] = data[f'{symbol}_MACD'].diff()
    data['Bollinger_Lower_Change'] = data[f'{symbol}_Bollinger_lband'].diff()
    data[f'{symbol}_Signal'] = ((data['MACD_Change'].abs() < 0.005) & 
                      (data['Bollinger_Lower_Change'] > 0.02)).astype(int)
    return data
```
8. 특성
``` python
buy_features = ['MACD', 'Bollinger_lband']
sell_features = ['EMA_5', 'SMA_5', 'ATR']
```
9. 양성/음성 샘플 중에 적은거 over sampling (Balanced Batch)