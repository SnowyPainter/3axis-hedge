import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ta import add_all_ta_features
from ta.trend import adx, cci
from ta.volume import on_balance_volume
from ta.momentum import rsi

def plot_technical_analysis(ohlcv, symbol, title_suffix):
    print(symbol)
    
    # 필요한 컬럼 추출 및 이름 설정
    ohlcv = ohlcv.rename(columns={
        f'{symbol}_Close': 'Close', 
        f'{symbol}_High': 'High',
        f'{symbol}_Low': 'Low', 
        f'{symbol}_Volume': 'Volume'
    })

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

    # 선택된 특성 리스트
    features = ['Close', 'MA20', 'MA50', 'RSI', 'VWAP', 'CMF', 'CCI', 'ADX', 'OBV']
    feature_data = ohlcv[features]

    # 1. 산점도 행렬
    plt.figure(figsize=(12, 10))
    sns.pairplot(feature_data, diag_kind="kde", corner=True)
    plt.suptitle(f"Scatterplot Matrix - {title_suffix}", fontsize=16)
    plt.savefig(f"harpoon-images/{symbol}_Scatterplot_Matrix_{title_suffix}.png")
    plt.close()

    # 2. 상관계수 히트맵
    corr = feature_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f"Correlation Heatmap - {title_suffix}", fontsize=16)
    plt.savefig(f"harpoon-images/{symbol}_Correlation_Heatmap_{title_suffix}.png")
    plt.close()

    # 3. Box Plot (특정 지표 확인)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=feature_data[['RSI', 'CCI', 'ADX']], palette='pastel')
    plt.title(f"Boxplot of Key Indicators - {title_suffix}", fontsize=16)
    plt.savefig(f"harpoon-images/{symbol}_Boxplot_Key_Indicators_{title_suffix}.png")
    plt.close()

    # 4. Time Series Plot
    plt.figure(figsize=(14, 7))
    plt.plot(ohlcv['Close'], label='Close Price', color='blue', alpha=0.8)
    plt.plot(ohlcv['VWAP'], label='VWAP', color='orange', linestyle='--', alpha=0.7)
    plt.plot(ohlcv['MA20'], label='MA20', color='green', linestyle='-.', alpha=0.7)
    plt.legend(loc='upper left')
    plt.title(f"Time Series Analysis - {title_suffix}", fontsize=16)
    plt.savefig(f"harpoon-images/{symbol}_Time_Series_Analysis_{title_suffix}.png")
    plt.close()


def analyze_high_volatility_and_plot(ohlcv, symbol, threshold=0.1):
    # 변동폭 계산
    ohlcv['Price_Change'] = (ohlcv[symbol+'_High'] - ohlcv[symbol+'_Low']) / ohlcv[symbol+'_Close'].shift(1)
    volatile_data = ohlcv[ohlcv['Price_Change'] > threshold]
    if not volatile_data.empty:
        for idx in volatile_data.index:
            start = max(0, ohlcv.index.get_loc(idx) - (90))
            end = min(len(ohlcv), ohlcv.index.get_loc(idx))
            surrounding_data = ohlcv.iloc[start:end]
            plot_technical_analysis(surrounding_data, symbol, title_suffix=f"High Volatility at {idx}")

if __name__ == "__main__":
    utils.create_pickle()
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
            #temp_df = calculate_technical_indicators(temp_df, stock)
            #for indicator in localbns.features:
            #    new_indicators[f'{stock}_{indicator}'] = temp_df[f'{stock}_{indicator}']

            analyze_high_volatility_and_plot(temp_df, stock, 0.3)