import sys, os
sys.path.append('../')

import utils
import pandas as pd
import glob

def create_pickle():
    crypto_list = pd.read_csv('./binance-cryptos.csv')

    crypto_symbols = crypto_list['BaseAsset'] + '-' + crypto_list['QuoteAsset']
    symbols = crypto_symbols.head(15)
    combined_df = pd.DataFrame()

    for symbol in symbols:
        df = utils.load_historical_for_learning(symbol, utils.today_before(8), utils.today(), interval='1m')
        if df.empty:
            continue

        if combined_df.empty:
            combined_df = df
        else:
            combined_df = combined_df.join(df, how='outer')

    combined_df.sort_index(inplace=True)
    original_columns = len(combined_df.columns)
    combined_df = combined_df.dropna(axis=1, how='all')
    removed_columns = original_columns - len(combined_df.columns)
    print(f"Removed {removed_columns} columns where all values were NaN.")
    print(f"Remaining columns: {len(combined_df.columns)}")
    combined_df.dropna(inplace=True)
    combined_df.to_pickle('./crypto-ohlcv.pkl')

def load_pickle():
    df = pd.read_pickle('./crypto-ohlcv.pkl')
    return df

def process_symbol_data(symbol, df_with_indicators, seq_length, features, target_function):
    print(f"Processing {symbol}...")
    data = df_with_indicators[[f'{symbol}_Open'] + [f'{symbol}_{feature}' for feature in features]].copy()
    data = target_function(data, symbol)
    
    data.dropna(inplace=True)
    
    symbol_X, symbol_y = [], []
    for i in range(len(data) - seq_length):
        symbol_X.append(data[[f'{symbol}_{feature}' for feature in features]].iloc[i:i+seq_length].values)
        symbol_y.append(data[f'{symbol}_Signal'].iloc[i+seq_length])
    
    print(f"{symbol} : Preprocessed")
    
    ones_count = sum(1 for i in symbol_y if i == 1)
    zeros_count = sum(1 for i in symbol_y if i == 0)
    twos_count = sum(1 for i in symbol_y if i == 2)
    print(f"Number of 0's: {zeros_count}, Number of 1's: {ones_count}, Number of 2's: {twos_count}")
    
    return symbol_X, symbol_y

def create_and_save_data(symbols, df_with_indicators, seq_length, features, target_function, prefix):
    X, y = [], []
    for symbol in symbols:
        symbol_X, symbol_y = process_symbol_data(symbol, df_with_indicators, seq_length, features, target_function)
        X.extend(symbol_X)
        y.extend(symbol_y)
        
        if len(X) > 4000:  # Adjust this threshold as needed
            utils.save_data_chunk(X, y, prefix)
            X, y = [], []
    
    if X:
        utils.save_data_chunk(X, y, prefix)

import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from sklearn.utils import class_weight
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from imblearn.combine import SMOTETomek
from collections import Counter

import ta

def evaluate_model(model, X_test, y_test):
    try:
        print("Evaluating trend model...")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.5f}")

        # 예측 및 추가 평가 지표
        y_pred = np.argmax(model.predict(X_test), axis=1)  # 다중 클래스일 경우
        y_true = np.argmax(y_test, axis=1)  # One-hot encoding일 경우

        # 분류 보고서 출력
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

    except Exception as e:
        print(f"Error during evaluation: {e}")

def oversample_data(X, y):
    """Applies SMOTE for oversampling."""
    n_samples, timesteps, n_features = X.shape
    X_reshaped = X.reshape(n_samples, timesteps * n_features)  # Flatten the data for SMOTE

    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
    
    # Reshape back to original dimensions
    X_resampled = X_resampled.reshape(-1, timesteps, n_features)
    
    return X_resampled, y_resampled

def train_model_with_oversampling(model, X_train, y_train):
    X_train_resampled, y_train_resampled = oversample_data(X_train, y_train)
    
    X_train_resampled = X_train_resampled.astype(np.float32)
    y_train_resampled = y_train_resampled.astype(np.float32)  # One-Hot 인코딩된 레이블
    
    print(f"Starting model training with oversampled data...")
    
    checkpoint = ModelCheckpoint(f'602o_univ.h5', monitor='val_loss', save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print(f"Starting 602o model training with oversampled data...")
    history = model.fit(
        X_train_resampled, y_train_resampled,
        epochs=50,
        batch_size=48,
        validation_split=0.2,
        callbacks=[checkpoint, early_stop]
    )
    return history

seq_length = 90

# 30으로 바꾸면 어떨까


prefix = 'CRYPTO'
features = ['MACD', 'Bollinger_lband', 'EMA_5', 'ATR', 'RSI','Volume_Change']

def _create_model(input_shape, loss='categorical_crossentropy'):
    model = Sequential([
        LSTM(128, activation='tanh', return_sequences=True, input_shape=input_shape),  # LSTM 유닛 수 증가
        Dropout(0.4),
        LSTM(64, activation='tanh', return_sequences=True),
        Dropout(0.4),
        LSTM(32, activation='tanh'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

def target_function(data, symbol, lookahead=5):
    # NaN 값 제거
    data = data.copy()
    data = data.dropna()

    data[f'{symbol}_MACD_Change'] = data[f'{symbol}_MACD'].diff(periods=lookahead)
    data[f'{symbol}_Bollinger_Lower_Change'] = data[f'{symbol}_Bollinger_lband'].diff(periods=lookahead)
    data[f'{symbol}_EMA_Change'] = data[f'{symbol}_EMA_5'].pct_change(periods=lookahead, fill_method=None)
    data[f'{symbol}_ATR_Change'] = data[f'{symbol}_ATR'].pct_change(periods=lookahead, fill_method=None)
    
    # lookahead 적용
    buy_signal = ((data[f'{symbol}_MACD_Change'].abs() < 0.03) & 
                    (data[f'{symbol}_Bollinger_Lower_Change'] < -0.001)).astype(int)
    
    sell_signal = ((data[f'{symbol}_EMA_Change'].abs() < 0.015) &
                    (data[f'{symbol}_ATR_Change'] > 0.005)).astype(int)
    
    data[f'{symbol}_Signal'] = 0  # 기본값: 변동 없음
    data.loc[buy_signal == 1, f'{symbol}_Signal'] = 1  # 극소점
    data.loc[sell_signal == 1, f'{symbol}_Signal'] = 2  # 극대점

    data = data.dropna()

    return data

def calculate_technical_indicators(df, symbol):
    df = df.copy()
    
    df[f'{symbol}_MACD'] = ta.trend.MACD(close=df[f'{symbol}_Close']).macd()
    bb_indicator = ta.volatility.BollingerBands(close=df[f'{symbol}_Close'])
    df[f'{symbol}_Bollinger_lband'] = bb_indicator.bollinger_lband()
    df[f'{symbol}_EMA_5'] = ta.trend.EMAIndicator(close=df[f'{symbol}_Close'], window=5).ema_indicator()
    df[f'{symbol}_ATR'] = ta.volatility.AverageTrueRange(high=df[f'{symbol}_High'], low=df[f'{symbol}_Low'], close=df[f'{symbol}_Close']).average_true_range()
    df[f'{symbol}_RSI'] = ta.momentum.RSIIndicator(close=df[f'{symbol}_Close']).rsi()
    df[f'{symbol}_Volume_Change'] = df[f'{symbol}_Volume'].pct_change(periods=5, fill_method=None)

    df.dropna(inplace=True)
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    scaler = StandardScaler()
    for feature in features:
        df[f'{symbol}_{feature}'] = scaler.fit_transform(df[[f'{symbol}_{feature}']])
    
    return df

def create_model(df_with_indicators, symbols):
    
    if not glob.glob(f'./chunks/{prefix}_data_chunk_*.pkl'):
        create_and_save_data(symbols, df_with_indicators, seq_length, features, target_function, f'{prefix}_')
    
    X_train, X_test, y_train, y_test = utils.load_and_split_data_onehot(f'{prefix}_')

    model = _create_model((seq_length, len(features)), loss='categorical_crossentropy')
    history = train_model_with_oversampling(model, X_train, y_train)
    print("Evaluating trend model...")
    
    model = load_model('602o_univ.h5')
    evaluate_model(model, X_test, y_test)
    
    return model

def _finetune_model(model, X, y, model_name, epochs=15, batch_size=32):
    checkpoint = ModelCheckpoint(f'best_{model_name}_finetuned_model.h5', monitor='loss', save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    print(f"Fine-tuning {model_name} model...")
    
    history = model.fit(
        X,
        y, 
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[checkpoint, early_stop]
    )
    return model, history

def finetune_model(symbol, df_with_indicators, original_model_path='602o_univ.h5'):
    symbol_data = df_with_indicators[[f'{symbol}_{feature}' for feature in features]].copy()
    symbol_data = target_function(symbol_data, symbol)
    
    X, y = [], []
    for i in range(len(symbol_data) - seq_length):
        X.append(symbol_data[[f'{symbol}_{feature}' for feature in features]].iloc[i:i+seq_length].values)
        y.append(symbol_data[f'{symbol}_Signal'].iloc[i+seq_length])
    
    X = np.array(X)
    y = np.array(y)

    class_counts = Counter(y)
    min_samples = min(class_counts.values())
    k_neighbors = min(max(1, min_samples - 1), 3)  # Ensure k_neighbors is <= min_samples - 1
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    smote_tomek = SMOTETomek(smote=smote, random_state=42)
    n_samples, timesteps, n_features = X.shape
    X_flat = X.reshape((n_samples, timesteps * n_features))
    
    X_resampled, y_resampled = smote_tomek.fit_resample(X_flat, y)
    encoder = OneHotEncoder(categories=[[0,1,2]], sparse_output=False)
    y_resampled = encoder.fit_transform(y_resampled.reshape(-1, 1))
    X_resampled = X_resampled.reshape((-1, timesteps, n_features))
    model = load_model(original_model_path)

    finetuned_model, history = _finetune_model(model, X_resampled, y_resampled, f'602o_{symbol}')
    
    return finetuned_model

def predict(model, raw, symbol):
    df = calculate_technical_indicators(raw, symbol).tail(seq_length)
    x = df[[f"{symbol}_{feature}" for feature in features]].values

    x = np.expand_dims(x, axis=0)
    return model.predict(x, verbose=0)[0]

if __name__ == "__main__":
    combined_prices = load_pickle()
    symbols = list(set(col.split('_')[0] for col in combined_prices.columns))
    
    if combined_prices is not None:
        df_with_indicators = combined_prices.copy()
        new_indicators = {}
        for stock in symbols:
            temp_df = pd.DataFrame({
                f'{stock}_Open': df_with_indicators[f"{stock}_Open"],
                f'{stock}_Close': df_with_indicators[f"{stock}_Close"],
                f'{stock}_High': df_with_indicators[f"{stock}_High"],
                f'{stock}_Low': df_with_indicators[f"{stock}_Low"],
                f'{stock}_Volume' : df_with_indicators[f"{stock}_Volume"]
            })
            temp_df = calculate_technical_indicators(temp_df, stock)
            for indicator in features:
                new_indicators[f'{stock}_{indicator}'] = temp_df[f'{stock}_{indicator}']
        df_with_indicators = pd.concat([df_with_indicators, pd.DataFrame(new_indicators)], axis=1)
        create_model(df_with_indicators, symbols)