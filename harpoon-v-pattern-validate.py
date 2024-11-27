import utils

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ta import add_all_ta_features
from ta.trend import adx, cci
from ta.volume import on_balance_volume
from ta.momentum import rsi
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def label_v_patterns(data, cmf_col='CMF', vwap_col='VWAP', window=90, slope_threshold=0.03):
    """
    CMF(y축)와 VWAP(x축)의 산점도에서 V자/역V자 패턴을 더욱 정밀하게 라벨링
    window 60일 단위로 패턴 탐지
    """
    data = data.copy()
    data['V_Pattern'] = 0  # 0: 패턴 없음, 1: V자, -1: 역V자
    
    for i in range(0, len(data)-window+1):
        window_data = data.iloc[i:i+window]
        
        if len(window_data) < window:
            continue
            
        cmf = window_data[cmf_col].values
        vwap = window_data[vwap_col].values
        
        # 최소점 및 최대점 찾기
        min_point = np.argmin(cmf)
        max_point = np.argmax(cmf)
        
        # 좌우 기울기 계산
        if min_point > 0 and min_point < len(cmf) - 1:
            left_slope = (cmf[min_point] - cmf[0]) / (vwap[min_point] - vwap[0] + 1e-6)
            right_slope = (cmf[-1] - cmf[min_point]) / (vwap[-1] - vwap[min_point] + 1e-6)
            
            # V자 패턴 확인
            if left_slope < -slope_threshold and right_slope > slope_threshold:
                data.loc[data.index[i + min_point], 'V_Pattern'] = 1  # V자형
            
            # 역V자 패턴 확인
            elif left_slope > slope_threshold and right_slope < -slope_threshold:
                data.loc[data.index[i + max_point], 'V_Pattern'] = -1  # 역V자형
            
    return data

def label_outcomes(data, window=30, threshold=0.2):
    """
    급등/급락 라벨링
    당일부터 3일 이내의 고가/저가 기준으로 급등/급락 판단
    """
    data = data.copy()

    high_3d = data['High'].rolling(window=window, min_periods=1).max()
    low_3d = data['Low'].rolling(window=window, min_periods=1).min()
    
    # 3일 이내 최고가/최저가 기준 수익률 계산
    high_returns = (high_3d - data['Open']) / data['Close'].shift(1)
    low_returns = (low_3d - data['Open']) / data['Close'].shift(1)
    
    # 급등/급락 라벨링
    data['Outcome'] = np.where(high_returns > threshold, 1,
                              np.where(low_returns < -threshold, -1, 0))  # 급락
    
    data['LR'] = low_returns
    data['HR'] = high_returns
    data['CR'] = (data['Close'].shift(-30) - data['Open']) / data['Close'].shift(1)

    return data

def prepare_dataset(data, feature_cols, label_col):
    """
    데이터셋 준비
    """
    data = data.dropna(subset=feature_cols + [label_col])
    X = data[feature_cols]
    y = data[label_col]
    return X, y

def train_model(X, y):
    """
    랜덤 포레스트를 사용한 모델 학습 및 평가
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 성능 평가
        y_pred = model.predict(X_test)
        results = pd.DataFrame({
            '실제결과': y_test,
            '예측결과': y_pred
        })
        accuracy = (results['실제결과'] == results['예측결과']).mean()
        print(f"모델 예측 정확도: {accuracy:.2%}")
        
        return model
    except ValueError:
        print("데이터 분할 중 오류 발생. 모델 학습을 건너뜁니다.")
        return None

def analyze_v_patterns(data, stock_name):
    """
    V자 패턴과 급등/급락 간의 상관관계 분석
    """
    v_pattern_df = data[data['V_Pattern'] == 1].dropna()  # V자형 패턴
    inverse_v_pattern_df = data[data['V_Pattern'] == -1].dropna()  # 역V자형 패턴
    v_surge_rate = (v_pattern_df['Outcome'] == 1).mean() if len(v_pattern_df) > 0 else 0
    v_plunge_rate = (v_pattern_df['Outcome'] == -1).mean() if len(v_pattern_df) > 0 else 0
    inverse_v_surge_rate = (inverse_v_pattern_df['Outcome'] == 1).mean() if len(inverse_v_pattern_df) > 0 else 0
    inverse_v_plunge_rate = (inverse_v_pattern_df['Outcome'] == -1).mean() if len(inverse_v_pattern_df) > 0 else 0

    return v_surge_rate, v_plunge_rate, inverse_v_surge_rate, inverse_v_plunge_rate


def analyze_all_stocks(combined_prices, threshold=0.2):
    """
    여러 주식에 대해 V자 패턴 분석 및 결과 종합
    """
    symbols = {col.split('_')[0] for col in combined_prices.columns}
    overall_results = []
    print(f"{threshold * 100 :.2f}% 등락 분석")
    for stock in symbols:
        print(f"분석 중: {stock}")
        temp_df = pd.DataFrame({
            f'{stock}_Open': combined_prices[f"{stock}_Open"],
            f'{stock}_High': combined_prices[f"{stock}_High"],
            f'{stock}_Low': combined_prices[f"{stock}_Low"],
            f'{stock}_Close': combined_prices[f"{stock}_Close"],
            f'{stock}_Volume': combined_prices[f"{stock}_Volume"]
        })
        ohlcv = temp_df.rename(columns={
            f'{stock}_Open': 'Open',
            f'{stock}_Close': 'Close', 
            f'{stock}_High': 'High',
            f'{stock}_Low': 'Low', 
            f'{stock}_Volume': 'Volume'
        })

        # 기술 지표 추가 및 라벨링
        df = utils.tech(ohlcv)
        df = label_v_patterns(df)
        df = label_outcomes(df, window=30, threshold=threshold)
        df.to_csv(f"harpoon-images/data/{stock}.csv")
        # 모델 학습
        features = ['MA20', 'MA50', 'RSI', 'VWAP', 'CMF', 'CCI', 'ADX', 'OBV']
        X, y = prepare_dataset(df, feature_cols=features, label_col="Outcome")
        if len(X) == 0:
            continue
        
        model = train_model(X, y)
        
        # V자 패턴 분석 및 결과 저장
        v_surge_rate, v_plunge_rate, inverse_v_surge_rate, inverse_v_plunge_rate = analyze_v_patterns(df, stock)
        overall_results.append({
            "stock": stock,
            "v_surge_rate": v_surge_rate,
            "v_plunge_rate": v_plunge_rate,
            "inverse_v_surge_rate": inverse_v_surge_rate,
            "inverse_v_plunge_rate": inverse_v_plunge_rate
        })
    
    # 전체 평균 계산 및 출력
    avg_v_surge = sum(r['v_surge_rate'] for r in overall_results) / len(overall_results)
    avg_v_plunge = sum(r['v_plunge_rate'] for r in overall_results) / len(overall_results)
    avg_inv_v_surge = sum(r['inverse_v_surge_rate'] for r in overall_results) / len(overall_results)
    avg_inv_v_plunge = sum(r['inverse_v_plunge_rate'] for r in overall_results) / len(overall_results)
    
    print(f"\n===== {threshold * 100 :.2f}% 등락 전체 평균 분석 결과 =====")
    print(f"전체 주식 V자 패턴 평균 급등 비율: {avg_v_surge:.2%}")
    print(f"전체 주식 V자 패턴 평균 급락 비율: {avg_v_plunge:.2%}")
    print(f"전체 주식 역V자 패턴 평균 급등 비율: {avg_inv_v_surge:.2%}")
    print(f"전체 주식 역V자 패턴 평균 급락 비율: {avg_inv_v_plunge:.2%}")

    return overall_results


if __name__ == "__main__":
    combined_prices = utils.load_combined_prices('sp500_combined_close_prices.pkl')
    if combined_prices is not None:
        thresholds = [0.2]
        for threshold in thresholds:
            analyze_all_stocks(combined_prices, threshold)
