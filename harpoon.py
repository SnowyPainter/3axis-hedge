import utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_technical_analysis(ohlcv, title_suffix):
    # 기술적 지표 계산
    ohlcv['MA20'] = ohlcv['Close'].rolling(window=20).mean()  # 20일 이동평균선
    ohlcv['MA50'] = ohlcv['Close'].rolling(window=50).mean()  # 50일 이동평균선
    ohlcv['RSI'] = 100 - (100 / (1 + (ohlcv['Close'].diff().clip(lower=0).rolling(window=14).mean() /
                                      ohlcv['Close'].diff().clip(upper=0).abs().rolling(window=14).mean())))
    ohlcv['Volume_Change'] = ohlcv['Volume'].pct_change()  # 거래량 변화율
    ohlcv['ATR'] = ohlcv['High'] - ohlcv['Low']  # True Range
    ohlcv['ATR'] = ohlcv['ATR'].rolling(window=14).mean()  # ATR 계산
    ohlcv['MACD'] = ohlcv['Close'].ewm(span=12).mean() - ohlcv['Close'].ewm(span=26).mean()  # MACD
    ohlcv['Signal_Line'] = ohlcv['MACD'].ewm(span=9).mean()  # MACD Signal Line

    # 결측값 제거
    ohlcv = ohlcv.dropna()

    # 기술적 특성만 선택
    features = ['Close', 'Volume', 'MA20', 'MA50', 'RSI', 'Volume_Change', 'ATR', 'MACD']
    feature_data = ohlcv[features]

    # 특성 간 산점도 행렬
    plt.figure(figsize=(12, 10))
    sns.pairplot(feature_data, diag_kind="kde", corner=True)
    plt.suptitle(f"Scatterplot Matrix - {title_suffix}", fontsize=16)
    plt.show()

def analyze_high_volatility_and_plot(ohlcv, threshold=0.1):
    # 변동폭 계산
    ohlcv['Price_Change'] = (ohlcv['High'] - ohlcv['Low']) / ohlcv['Close'].shift(1)
    volatile_data = ohlcv[ohlcv['Price_Change'] > threshold]

    if not volatile_data.empty:
        for idx in volatile_data.index:
            # 이전 5개 데이터와 이후 5개 데이터 추출
            start = max(0, ohlcv.index.get_loc(idx) - (24*7))
            end = min(len(ohlcv), ohlcv.index.get_loc(idx) + 1)
            surrounding_data = ohlcv.iloc[start:end]
            print(surrounding_data[['Close', 'High', 'Low', 'Volume', 'Price_Change']])

            # 주변 데이터로 기술적 분석 시각화
            plot_technical_analysis(surrounding_data, title_suffix=f"High Volatility at {idx}")

# Main 실행
gy = "006050.KQ"
intervals = ['1h']  # '1d' 데이터는 제거

for interval in intervals:
    whale = utils.get_OHLCV(gy, "2023-01-01", "2024-11-24", interval=interval)

    # 10% 이상의 변동폭에 대해 분석 및 플롯
    analyze_high_volatility_and_plot(whale, threshold=0.1)
