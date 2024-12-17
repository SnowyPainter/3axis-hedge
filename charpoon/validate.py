import sys, os
sys.path.append('../')

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
from sklearn.preprocessing import MinMaxScaler
import os
import seaborn as sns

def label_v_patterns(data, cmf_col='CMF', vwap_col='VWAP', window=90, slope_threshold=0.01):
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

def label_v2_patterns(data, obv_col='OBV', vwap_col='VWAP', window=90, slope_threshold=0.01):
    """
    OBV(y축)와 VWAP(x축)의 산점도에서 V2 패턴 탐지 및 라벨링
    - 1: 상승(양의 상관관계 또는 역 V자 패턴)
    - 2: 하강(음의 상관관계)
    - 0: 패턴 없음
    """
    data = data.copy()
    data['V2_Pattern'] = 0  # 0: 패턴 없음, 1: 상승, 2: 하강

    for i in range(0, len(data) - window + 1):
        # 현재 window에 해당하는 데이터
        window_data = data.iloc[i:i + window]
        
        if len(window_data) < window:
            continue

        obv = window_data[obv_col].values
        vwap = window_data[vwap_col].values

        # 최소점 및 최대점 탐색
        min_point = np.argmin(obv)
        max_point = np.argmax(obv)

        # 역 V자 조건 확인
        if min_point > 0 and min_point < len(obv) - 1:
            left_slope = (obv[min_point] - obv[0]) / (vwap[min_point] - vwap[0] + 1e-6)
            right_slope = (obv[-1] - obv[min_point]) / (vwap[-1] - vwap[min_point] + 1e-6)
            if left_slope > slope_threshold and right_slope < -slope_threshold:
                data.loc[data.index[i:i + window], 'V2_Pattern'] = 1  # 역 V자
            
        # 선형 회귀를 통한 일반적 상승/하강 탐지
        else:
            slope, _ = np.polyfit(vwap, obv, 1)  # 1차 회귀선

            if slope > slope_threshold:  # 상승
                data.loc[data.index[i:i + window], 'V2_Pattern'] = 1
            elif slope < -slope_threshold:  # 하강
                data.loc[data.index[i:i + window], 'V2_Pattern'] = 2

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
    v2_pattern_up_df = data[data['V2_Pattern'] == 1].dropna()  # v2 상승 패턴
    v2_pattern_down_df = data[data['V2_Pattern'] == 2].dropna()  # v2 하강 패턴

    v_surge_rate = (v_pattern_df['Outcome'] == 1).mean() if len(v_pattern_df) > 0 else 0
    v_plunge_rate = (v_pattern_df['Outcome'] == -1).mean() if len(v_pattern_df) > 0 else 0
    inverse_v_surge_rate = (inverse_v_pattern_df['Outcome'] == 1).mean() if len(inverse_v_pattern_df) > 0 else 0
    inverse_v_plunge_rate = (inverse_v_pattern_df['Outcome'] == -1).mean() if len(inverse_v_pattern_df) > 0 else 0
    
    v2_up_surge_rate = (v2_pattern_up_df['CR'] > 0).mean() if len(v2_pattern_up_df) > 0 else 0
    v2_up_plunge_rate = (v2_pattern_up_df['CR'] < 0).mean() if len(v2_pattern_up_df) > 0 else 0
    v2_down_surge_rate = (v2_pattern_down_df['CR'] > 0).mean() if len(v2_pattern_down_df) > 0 else 0
    v2_down_plunge_rate = (v2_pattern_down_df['CR'] < 0).mean() if len(v2_pattern_down_df) > 0 else 0
    
    return {
        "v_급등_비율": v_surge_rate,
        "v_급락_비율": v_plunge_rate,
        "역v_급등_비율": inverse_v_surge_rate,
        "역v_급락_비율": inverse_v_plunge_rate,
        "v2_상승_급등_비율": v2_up_surge_rate,
        "v2_상승_급락_비율": v2_up_plunge_rate,
        "v2_하강_급등_비율": v2_down_surge_rate,
        "v2_하강_급락_비율": v2_down_plunge_rate
    }

def save_image(df, stock):
    for outcome_type, outcome_val in [("Down", -1), ("Up", 1)]:
        outcome_indices = df[df['Outcome'] == outcome_val].index
        stock_dir = f"harpoon-images/{stock}"
        outcome_dir = f"{stock_dir}/{outcome_type}"
        os.makedirs(outcome_dir, exist_ok=True)
        
        for idx in np.random.choice(outcome_indices, min(2, len(outcome_indices)), replace=False):
            if idx >= df.index[90]:
                prev_data = df.loc[idx - pd.Timedelta(days=90):idx]
                plt.figure(figsize=(12, 12))
                features = ['MA20', 'RSI', 'VWAP', 'CMF', 'CCI', 'ADX', 'OBV']
                sns.pairplot(prev_data[features])
                plt.tight_layout()
                plt.savefig(f"{outcome_dir}/scatter_{pd.Timestamp(idx).strftime('%Y%m%d')}.png")
                plt.close()
                
                plt.figure(figsize=(12, 6))
                scaler = MinMaxScaler()
                scaled_data = pd.DataFrame(scaler.fit_transform(prev_data[['Close', 'High', 'Low', 'VWAP', 'CMF']]), 
                                        columns=['Close', 'High', 'Low', 'VWAP', 'CMF'],
                                        index=prev_data.index)
                plt.plot(scaled_data.index, scaled_data['Close'], label='Close Price')
                plt.plot(scaled_data.index, scaled_data['High'], label='High Price') 
                plt.plot(scaled_data.index, scaled_data['Low'], label='Low Price')
                plt.plot(scaled_data.index, scaled_data['VWAP'], label='MA20')
                plt.plot(scaled_data.index, scaled_data['CMF'], label='MA50')
                plt.title(f"{stock} Price Chart - {outcome_type} at {pd.Timestamp(idx).strftime('%Y-%m-%d')}")
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{outcome_dir}/price_{pd.Timestamp(idx).strftime('%Y%m%d')}.png")
                plt.close()

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
        df = label_v_patterns(df, window=72, slope_threshold=30)
        df = label_v2_patterns(df, window=72, slope_threshold=2)
        df = label_outcomes(df, window=36, threshold=threshold)

        #save_image(df, stock)
        
        df.to_csv(f"csvs/{stock}.csv")
        # 모델 학습
        features = ['MA20', 'MA50', 'RSI', 'VWAP', 'CMF', 'CCI', 'ADX', 'OBV']
        X, y = prepare_dataset(df, feature_cols=features, label_col="Outcome")
        if len(X) == 0:
            continue
        
        #model = train_model(X, y)
        # V자 패턴 분석 및 결과 저장
        v_pattern_results = analyze_v_patterns(df, stock)
        overall_results.append({
            "stock": stock,
            **v_pattern_results
        })
    
    # 전체 평균 계산 및 출력
    avg_results = {key: sum(r[key] for r in overall_results) / len(overall_results) 
                   for key in v_pattern_results.keys()}
    
    print(f"\n===== {threshold * 100 :.2f}% 등락 전체 평균 분석 결과 =====")
    for key, value in avg_results.items():
        print(f"전체 주식 {key} 평균 비율: {value:.2%}")

    return overall_results


if __name__ == "__main__":
    combined_prices = utils.load_combined_prices('./crypto-ohlcv.pkl')
    if combined_prices is not None:
        thresholds = [0.1]
        for threshold in thresholds:
            analyze_all_stocks(combined_prices, threshold)
