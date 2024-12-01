import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ta import add_all_ta_features
from ta.trend import adx, cci
from ta.volume import on_balance_volume
from ta.momentum import rsi
from sklearn.preprocessing import MinMaxScaler


def tech(ohlcv):
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
    return ohlcv

def plot_technical_analysis(ohlcv, ohlcv2, point, symbol, title_suffix):
    # 필요한 컬럼 추출 및 이름 설정

    ohlcv = ohlcv.rename(columns={
        f'{symbol}_Open': 'Open',
        f'{symbol}_Close': 'Close', 
        f'{symbol}_High': 'High',
        f'{symbol}_Low': 'Low', 
        f'{symbol}_Volume': 'Volume'
    })

    ohlcv2 = ohlcv2.rename(columns={
        f'{symbol}_Open': 'Open',
        f'{symbol}_Close': 'Close', 
        f'{symbol}_High': 'High',
        f'{symbol}_Low': 'Low', 
        f'{symbol}_Volume': 'Volume'
    })

    # 추가 기술적 지표 계산
    ohlcv = tech(ohlcv)
    ohlcv2 = tech(ohlcv2)
    scaler = MinMaxScaler()
    ohlcv2[['Close', 'MA20', 'MA50', 'RSI', 'VWAP', 'CMF', 'CCI', 'ADX', 'OBV']] = scaler.fit_transform(ohlcv2[['Close', 'MA20', 'MA50', 'RSI', 'VWAP', 'CMF', 'CCI', 'ADX', 'OBV']])

    # 결측값 제거
    ohlcv = ohlcv.dropna()

    # 선택된 특성 리스트
    features = ['Close', 'MA20', 'MA50', 'RSI', 'VWAP', 'CMF', 'CCI', 'ADX', 'OBV']
    feature_data = ohlcv[features]

    import os
    import mplfinance as mpf
    # title_suffix에 따른 폴더 생성
    folder_name = f"harpoon-images/{title_suffix}"
    os.makedirs(folder_name, exist_ok=True)

    # 1. 산점도 행렬
    plt.figure(figsize=(12, 10))
    sns.pairplot(feature_data, diag_kind="kde", corner=True)
    plt.suptitle(f"산점도 행렬 - {title_suffix}", fontsize=16)
    plt.savefig(f"{folder_name}/{symbol}_산점도_행렬.png")
    plt.close()

    # 2. 상관계수 히트맵
    corr = feature_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f"상관계수 히트맵 - {title_suffix}", fontsize=16)
    plt.savefig(f"{folder_name}/{symbol}_상관계수_히트맵.png")
    plt.close()

    # 4. 일반 차트와 기술적 지표
    plt.figure(figsize=(14, 10))
    
    # 주가 데이터 플롯
    plt.plot(ohlcv2.index, ohlcv2['Close'], label='종가', color='black')
    plt.scatter(point, ohlcv2.loc[point, 'Close'], color='red', s=100, zorder=5, label='포착 시점')
    plt.plot(ohlcv2.index, ohlcv2['CCI'], label='CCI', color='blue', alpha=0.7)
    plt.plot(ohlcv2.index, ohlcv2['MA50'], label='MA50', color='red', alpha=0.7)
    plt.plot(ohlcv2.index, ohlcv2['VWAP'], label='VWAP', color='green', alpha=0.7)
    plt.title(f"주가 차트 - {title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{folder_name}/{symbol}_주가_차트.png")
    plt.close()


def analyze_high_volatility_and_plot(ohlcv, symbol, threshold=0.1):
    # 변동폭 계산
    
    s = "급등"
    if threshold < 0:
        s = "급락"
        ohlcv['Price_Change'] = (ohlcv[symbol+'_Low'] - ohlcv[symbol+'_Open']) / ohlcv[symbol+'_Close'].shift(1)
        volatile_data = ohlcv[ohlcv['Price_Change'] <= threshold]
    else:
        ohlcv['Price_Change'] = (ohlcv[symbol+'_High'] - ohlcv[symbol+'_Open']) / ohlcv[symbol+'_Close'].shift(1)
        volatile_data = ohlcv[ohlcv['Price_Change'] >= threshold]
    if not volatile_data.empty:
        for idx in volatile_data.index:
            start = max(0, ohlcv.index.get_loc(idx) - (90))
            end = min(len(ohlcv), ohlcv.index.get_loc(idx))
            end2 = min(len(ohlcv), ohlcv.index.get_loc(idx) + 10)
            surrounding_data = ohlcv.iloc[start:end]

            plot_technical_analysis(surrounding_data, ohlcv[start:end2], idx, symbol, 
                title_suffix=f"{s} {symbol} {idx}")

if __name__ == "__main__":
    #utils.create_pickle()
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

            analyze_high_volatility_and_plot(temp_df, stock, threshold=-0.2)