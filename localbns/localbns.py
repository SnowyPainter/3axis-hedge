from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import ta 

seqlen = 100

def nplog(df):
    data_log = df.apply(lambda x: np.log(x + 1))
    return data_log

def normalize(df):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

from sklearn.preprocessing import MinMaxScaler

def min_max_scale(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)


def create_sequences(data, features, target, seq_length):
    print(data)
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[features].iloc[i:i+seq_length].values)
        y.append(data[target].iloc[i+seq_length])
    return np.array(X), np.array(y)

def get_52_ba(df, symbol, max_attempts=20):
    def find_max_price(df, symbol, attempt):
        max_price_date = df[symbol + '_Price'].idxmax()
        max_price_idx = df.index.get_loc(max_price_date)
        half_size = min(max_price_idx, len(df) - max_price_idx - 1)
        if half_size < 100 and attempt < max_attempts:
            df_exclude_max = df[df.index != max_price_date]
            return find_max_price(df_exclude_max, symbol, attempt + 1)
        df_before = df.iloc[max_price_idx - half_size:max_price_idx + 1]
        df_after = df.iloc[max_price_idx:max_price_idx + half_size + 1]
        
        return df_before, df_after

    return find_max_price(df, symbol, attempt=1)

features = ['SMA_5', 'EMA_5', 'RSI', 'MACD', 'Bollinger_hband', 'Bollinger_lband', 'ATR', 'Volume_Change']
def calculate_technical_indicators(df, symbol):
    df = df.copy()
    
    df.loc[:, f'{symbol}_SMA_5'] = ta.trend.SMAIndicator(df[symbol+'_Price'], window=5).sma_indicator()
    df.loc[:, f'{symbol}_EMA_5'] = ta.trend.EMAIndicator(df[symbol+'_Price'], window=5).ema_indicator()
    df.loc[:, f'{symbol}_RSI'] = ta.momentum.RSIIndicator(df[symbol+'_Price'], window=14).rsi()
    df.loc[:, f'{symbol}_MACD'] = ta.trend.MACD(df[symbol+'_Price']).macd()
    bollinger = ta.volatility.BollingerBands(df[symbol+'_Price'])
    df.loc[:, f'{symbol}_Bollinger_hband'] = bollinger.bollinger_hband()
    df.loc[:, f'{symbol}_Bollinger_lband'] = bollinger.bollinger_lband()
    df.loc[:, f'{symbol}_ATR'] = ta.volatility.AverageTrueRange(df[symbol+'_High'], df[symbol+'_Low'], df[symbol+'_Price'], window=14).average_true_range()
    df[f'{symbol}_Volume_Change'] = df[symbol+'_Volume'].pct_change().fillna(0)
    df.dropna(inplace=True)
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    # 각 지표를 정규화
    for feature in features:
        df[f'{symbol}_{feature}'] = (df[f'{symbol}_{feature}'] - df[f'{symbol}_{feature}'].mean()) / df[f'{symbol}_{feature}'].std()
    
    return df

def _calculate_macd(df, symbol):
        df['EMA12'] = df[symbol+"_Price"].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df[symbol+"_Price"].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        return df

def _calculate_bollinger_bands(df, symbol, window=20, num_sd=2):
    df['SMA20'] = df[symbol+"_Price"].rolling(window=window).mean()
    df['STD20'] = df[symbol+"_Price"].rolling(window=window).std()
    df['Bollinger_Upper'] = df['SMA20'] + (df['STD20'] * num_sd)
    df['Bollinger_Lower'] = df['SMA20'] - (df['STD20'] * num_sd)
    return df

def _calculate_ema(df, symbol, span=5):
    df[f'EMA_{span}'] = df[symbol+'_Price'].ewm(span=span, adjust=False).mean()
    return df
def _calculate_sma(df, symbol, window=5):
    df[f'SMA_{window}'] = df[symbol+'_Price'].rolling(window=window).mean()
    return df
def _calculate_atr(df, symbol, window=14):
    df['High-Low'] = df[symbol+'_High'] - df[symbol+'_Low']
    df['High-Close'] = np.abs(df[symbol+'_High'] - df[symbol+'_Price'].shift())
    df['Low-Close'] = np.abs(df[symbol+'_Low'] - df[symbol+'_Price'].shift())
    df['TR'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=window).mean()
    return df

def predict(raw, symbol, buy_model=None, sell_model=None):
    tech = calculate_technical_indicators(raw, symbol).tail(seqlen)
    
    buy_x = tech[[f'{symbol}_MACD', f'{symbol}_Bollinger_lband']].values
    sell_x = tech[[f'{symbol}_EMA_5', f'{symbol}_ATR']].values
    
    buy_x = np.expand_dims(buy_x, axis=0)
    sell_x = np.expand_dims(sell_x, axis=0)
    
    buy_y, sell_y = None, None
    if buy_model is not None:
        buy_y = buy_model.predict(buy_x, verbose=0)[0][0]
    if sell_model is not None:
        sell_y = sell_model.predict(sell_x, verbose=0)[0][0]
    return buy_y, sell_y

def create_buy_model(raw, symbol):
    
    '''
    신고가 갱신 이전 raw
    관찰대로라면, MACD의 변화율이 미미한 상황에서 볼린저 밴드 하단이 크게 요동치는 경우 이때의 매수 타이밍이 최적이다. 이를 계량적으로 판단하기 위해서는 기계학습이 동원될 필요가 있다. MACD의 변동폭이 줄어들고 볼린저 밴드 하단이 하락하기 시작하면 최적의 매수 타이밍으로 간주함으로써 모델을 구축할 수 있다.
    '''
    
    raw = normalize(raw)
    data = _calculate_macd(raw, symbol)
    data = _calculate_bollinger_bands(raw, symbol)
    data['MACD_Change'] = data['MACD'].diff()
    data['Bollinger_Lower_Change'] = data['Bollinger_Lower'].diff()
    data['Buy_Signal'] = ((data['MACD_Change'].abs() < 0.001) & 
                          (data['Bollinger_Lower_Change'] > 0.005)).astype(int)
    
    data.dropna(inplace=True)
    seq_length = seqlen
    features = ['MACD', 'Bollinger_Lower']
    target = 'Buy_Signal'
    X, y = create_sequences(data, features, target, seq_length)
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=(seq_length, len(features))),
        Dropout(0.3),
        LSTM(24, activation='tanh', return_sequences=False),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=300, batch_size=48, validation_split=0.2, verbose=1)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss : {loss :.5f}, Accuracy: {accuracy :.5f}")
    return model

def create_sell_model(raw, symbol):
    
    '''
    신고가 갱신 이후 raw
    관찰대로라면, 이동 평균의 변화가 미미하고 ATR이 상승하는 상황이라면 한 번의 하락 이후 다음 상승에서 매도하는 것이 최적이다. 이를 계량적으로 판단하기 위해서는 강화학습을 통해 패턴 후 매도하는것도 좋지만, ATR이 상승할 때 매도하고, 재진입을 시도하는 것이 보다 이득이다.
    '''
    
    raw = nplog(raw)
    data = _calculate_ema(raw, symbol, 5)
    data = _calculate_sma(raw, symbol, 5)
    data = _calculate_atr(raw, symbol, 14)
    data['EMA_Change'] = data['EMA_5'].diff()
    data['SMA_Change'] = data['SMA_5'].diff()
    data['ATR_Change'] = data['ATR'].diff()
    data['Sell_Signal'] = ((data['EMA_Change'].abs() < 0.005) &
                        (data['SMA_Change'].abs() < 0.005) &
                        (data['ATR_Change'] > 0.003)).astype(int)
    data.dropna(inplace=True)
    seq_length = seqlen
    features = ['EMA_5', 'SMA_5', 'ATR']
    target = 'Sell_Signal'
    X, y = create_sequences(data, features, target, seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=(seq_length, len(features))),
        Dropout(0.3),
        LSTM(24, activation='tanh', return_sequences=False),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=24, validation_split=0.2, verbose=1)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss : {loss :.5f}, Accuracy: {accuracy :.5f}")
    return model
    