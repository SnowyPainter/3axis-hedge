import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from ta import add_all_ta_features
from ta.trend import adx, cci
from ta.volume import on_balance_volume
from ta.momentum import rsi

def calculate_slopes(data, feature, window=5):
    """
    특정 지표의 기울기 계산 (V자형/역V자형 분석용)
    """
    slopes = []
    for i in range(len(data)):
        if i < window or i > len(data) - window - 1:
            slopes.append(np.nan)  # 가장자리는 계산 불가
        else:
            left_slope = (data[feature].iloc[i] - data[feature].iloc[i - window]) / window
            right_slope = (data[feature].iloc[i + window] - data[feature].iloc[i]) / window
            slopes.append((left_slope, right_slope))
    return slopes

def label_v_patterns(ohlcv, cmf_col, obv_col, ma_col, vwap_col, threshold=0.1):
    """
    V자형/역V자형 패턴 라벨링
    """
    ohlcv = ohlcv.copy()  # 슬라이스 작업 후 copy 생성
    # 추가 기술적 지표 계산
    ohlcv['MA20'] = ohlcv['Close'].rolling(window=20).mean()
    ohlcv['MA50'] = ohlcv['Close'].rolling(window=50).mean()
    ohlcv['RSI'] = rsi(ohlcv['Close'], window=14)
    ohlcv['VWAP'] = (ohlcv['Close'] * ohlcv['Volume']).cumsum() / ohlcv['Volume'].cumsum()
    ohlcv['CMF'] = ((ohlcv['Close'] - ohlcv['Low']) - (ohlcv['High'] - ohlcv['Close'])) / \
                   (ohlcv['High'] - ohlcv['Low']) * ohlcv['Volume']
    ohlcv['CMF'] = ohlcv['CMF'].rolling(window=20).mean()
    ohlcv['CCI'] = cci(ohlcv['High'], ohlcv['Low'], ohlcv['Close'], window=20)
    ohlcv['ADX'] = adx(ohlcv['High'], ohlcv['Low'], ohlcv['Close'], window=14)
    ohlcv['OBV'] = on_balance_volume(ohlcv['Close'], ohlcv['Volume'])

    # 결측값 제거
    ohlcv = ohlcv.dropna()

    ohlcv['Slopes'] = calculate_slopes(ohlcv, cmf_col)
    ohlcv['Left_Slope'] = ohlcv['Slopes'].apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
    ohlcv['Right_Slope'] = ohlcv['Slopes'].apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)
    
    # V자형 조건
    ohlcv['Label'] = np.where(
        (ohlcv['Left_Slope'] < -threshold) & (ohlcv['Right_Slope'] > threshold), 1,  # V자형
        np.where(
            (ohlcv['Left_Slope'] > threshold) & (ohlcv['Right_Slope'] < -threshold), 0,  # 역V자형
            np.nan
        )
    )
    return ohlcv.dropna(subset=['Label'])

def prepare_training_data(ohlcv, features, label_col):
    """
    학습 데이터 준비
    """
    X = ohlcv[features].dropna()
    y = ohlcv[label_col].loc[X.index]
    return X, y

def train_v_pattern_model(X, y):
    """
    머신러닝 모델 학습 및 평가
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 성능 평가
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return model

import utils
if __name__ == "__main__":
    combined_prices = utils.load_combined_prices('sp500_combined_close_prices.pkl')
    symbols = [col for col in combined_prices.columns if '_' not in col]
    if combined_prices is not None:
        df_with_indicators = combined_prices.copy()
        new_indicators = {}
        symbols = set()
        for col in combined_prices.columns:
            symbols.add(col.split('_')[0])
        for stock in symbols:
            temp_df = pd.DataFrame({
                f'{stock}_Open': df_with_indicators[f"{stock}_Open"],
                f'{stock}_High': df_with_indicators[f"{stock}_High"],
                f'{stock}_Low': df_with_indicators[f"{stock}_Low"],
                f'{stock}_Close': df_with_indicators[f"{stock}_Close"],
                f'{stock}_Volume' : df_with_indicators[f"{stock}_Volume"]
            })
            ohlcv = temp_df.rename(columns={
                f'{stock}_Open': 'Open',
                f'{stock}_Close': 'Close', 
                f'{stock}_High': 'High',
                f'{stock}_Low': 'Low', 
                f'{stock}_Volume': 'Volume'
            })

            features = ["CMF", "OBV", "MA20", "VWAP"]
            ohlcv = label_v_patterns(ohlcv, cmf_col="CMF", obv_col="OBV", ma_col="MA20", vwap_col="VWAP")
            X, y = prepare_training_data(ohlcv, features, label_col="Label")
            model = train_v_pattern_model(X, y)
            print(f"{stock}의 특성 중요도:")
            for feature, importance in zip(features, model.feature_importances_):
                print(f"{feature}: {importance}")
            print("\n")