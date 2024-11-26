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


def calculate_slope(data, feature, window=30):
    """
    주어진 윈도우 크기를 기준으로 기울기를 계산합니다.
    왼쪽 기울기(left)는 윈도우의 앞부분을, 오른쪽 기울기(right)는 윈도우의 뒷부분을 기준으로 계산합니다.
    """
    slopes_left = []
    slopes_right = []
    
    for i in range(len(data) - window + 1):
        segment_left = data[feature].iloc[i:i + window // 2]  # 왼쪽 절반
        segment_right = data[feature].iloc[i + window // 2:i + window]  # 오른쪽 절반
        
        # 왼쪽 기울기
        slope_left = (segment_left.iloc[-1] - segment_left.iloc[0]) / (len(segment_left) - 1)
        slopes_left.append(slope_left)
        
        # 오른쪽 기울기
        slope_right = (segment_right.iloc[-1] - segment_right.iloc[0]) / (len(segment_right) - 1)
        slopes_right.append(slope_right)
    
    # 기울기 계산 결과를 NaN으로 채운 나머지 부분
    slopes_left = [np.nan] * (window // 2) + slopes_left
    slopes_right = [np.nan] * (window // 2) + slopes_right
    
    return slopes_left, slopes_right

def label_v_patterns(data, cmf_col='CMF', vwap_col='VWAP', window=5, threshold=0.001):
    """
    CMF와 VWAP에 대해 V자형/역V자형 패턴을 라벨링
    """
    # 기울기 계산
    slopes_cmf_l, slopes_cmf_r = calculate_slope(data, feature=cmf_col, window=window)
    slopes_vwap_l, slopes_vwap_r = calculate_slope(data, feature=vwap_col, window=window)
    data = data.copy()
    # 기울기 데이터 추가
    data = data.tail(len(slopes_vwap_r))
    data['CMF_Left_Slope'] = slopes_cmf_l
    data['CMF_Right_Slope'] = slopes_cmf_r
    data['VWAP_Left_Slope'] = slopes_vwap_l
    data['VWAP_Right_Slope'] = slopes_vwap_r
    slopes = data[['CMF_Left_Slope', 'CMF_Right_Slope', 'VWAP_Left_Slope', 'VWAP_Right_Slope']]
    # 스케일링 적용
    scaler = StandardScaler()
    scaled_slopes = scaler.fit_transform(slopes)

    # 스케일링된 슬로프 데이터를 다시 데이터프레임에 저장
    data['CMF_Left_Slope'] = scaled_slopes[:, 0]
    data['CMF_Right_Slope'] = scaled_slopes[:, 1]
    data['VWAP_Left_Slope'] = scaled_slopes[:, 2]
    data['VWAP_Right_Slope'] = scaled_slopes[:, 3]
    data.dropna(inplace=True)
    '''
    sample_idx = range(50, 100)  # 데이터의 일부분

    plt.plot(data['CMF_Left_Slope'][sample_idx], label='CMF Left Slope', color='blue')
    plt.plot(data['CMF_Right_Slope'][sample_idx], label='CMF Right Slope', color='orange')
    plt.plot(data['VWAP_Left_Slope'][sample_idx], label='VWAP Left Slope', color='green')
    plt.plot(data['VWAP_Right_Slope'][sample_idx], label='VWAP Right Slope', color='red')
    plt.legend()
    plt.title("Slopes over sample index")
    plt.show()
    '''
    # V자형/역V자형 라벨링 (CMF + VWAP 조건)
    # 기울기 차이가 일정 비율 이상일 때 V자형/역V자형 패턴을 찾아냄
    threshold_scaled = 0.01
    data['V_Pattern'] = np.where(
        (data['CMF_Left_Slope'] < -threshold_scaled) & 
        (data['CMF_Right_Slope'] > threshold_scaled) &
        (data['VWAP_Left_Slope'] < -threshold_scaled) & 
        (data['VWAP_Right_Slope'] > threshold_scaled), 1,  # V자형
        np.where(
            (data['CMF_Left_Slope'] > threshold_scaled) & 
            (data['CMF_Right_Slope'] < -threshold_scaled) &
            (data['VWAP_Left_Slope'] > threshold_scaled) & 
            (data['VWAP_Right_Slope'] < -threshold_scaled), 0,  # 역V자형
            np.nan
        )
    )
    return data.dropna(subset=['V_Pattern'])

def label_outcomes(data, window=30, threshold=0.2):
    """
    급등/급락 라벨링
    """
    future_returns = (data['Close'].shift(-window) - data['Close']) / data['Close']
    data['Outcome'] = np.where(future_returns > threshold, 1,  # 급등
                               np.where(future_returns < -threshold, 0, np.nan))  # 급락
    return data.dropna(subset=['Outcome'])

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 성능 평가
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model

def analyze_v_patterns(data, stock_name):
    """
    V자 패턴과 급등/급락 간의 상관관계 분석
    """
    v_pattern_df = data[data['V_Pattern'] == 1]  # V자형 패턴
    inverse_v_pattern_df = data[data['V_Pattern'] == 0]  # 역V자형 패턴

    # V자 패턴 발생 시 급등 비율
    v_success_rate = (v_pattern_df['Outcome'] == 1).mean()
    
    # 역V자 패턴 발생 시 급락 비율
    inverse_v_success_rate = (inverse_v_pattern_df['Outcome'] == 0).mean()
    
    # 패턴별 Outcome 분포
    print(f"\n[{stock_name}] 분석 결과")
    print(f"V자 패턴 성공률 (급등): {v_success_rate:.2%}")
    print(f"역V자 패턴 성공률 (급락): {inverse_v_success_rate:.2%}")
    
    print("\nV자형 패턴 Outcome 분포:")
    print(v_pattern_df['Outcome'].value_counts(normalize=True))
    
    print("\n역V자형 패턴 Outcome 분포:")
    print(inverse_v_pattern_df['Outcome'].value_counts(normalize=True))
    
    return v_success_rate, inverse_v_success_rate


def analyze_all_stocks(combined_prices):
    """
    여러 주식에 대해 V자 패턴 분석 및 결과 종합
    """
    symbols = {col.split('_')[0] for col in combined_prices.columns}
    overall_results = []

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
        df = label_outcomes(df, window=30, threshold=0.2)

        # 모델 학습
        features = ['MA20', 'MA50', 'RSI', 'VWAP', 'CMF', 'CCI', 'ADX', 'OBV']
        X, y = prepare_dataset(df, feature_cols=features, label_col="Outcome")
        if len(X) == 0:
            continue
        
        model = train_model(X, y)

        # 중요도 출력
        print(f"\n[Feature Importances for {stock}]")
        for feature, importance in zip(features, model.feature_importances_):
            print(f"{feature}: {importance:.4f}")

        # V자 패턴 분석 및 결과 저장
        v_success_rate, inverse_v_success_rate = analyze_v_patterns(df, stock)
        overall_results.append({
            "stock": stock,
            "v_success_rate": v_success_rate,
            "inverse_v_success_rate": inverse_v_success_rate
        })

    # 전체 결과 요약
    print("\n===== 전체 주식 분석 결과 =====")
    for result in overall_results:
        print(f"{result['stock']}: V자 성공률 {result['v_success_rate']:.2%}, 역V자 성공률 {result['inverse_v_success_rate']:.2%}")

    return overall_results


if __name__ == "__main__":
    combined_prices = utils.load_combined_prices('sp500_combined_close_prices.pkl')
    if combined_prices is not None:
        analyze_all_stocks(combined_prices)
