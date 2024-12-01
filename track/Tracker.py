import os
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from keras.regularizers import l2
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle
from imblearn.over_sampling import SMOTE
import utils
import localbns.localbns as localbns

def create_pickle(directory='./stock_market_data/sp500/', name = 'sp500_combined_close_prices.pkl'):
    top_companies = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'GOOG', 'TSLA',
        'UNH', 'JPM', 'JNJ', 'V', 'XOM', 'PG', 'MA', 'LLY', 'HD', 'AVGO', 'CVX',
        'ABBV', 'MRK', 'PEP', 'KO', 'BAC'
    ]

    combined_df, _ = utils.load_historical_datas(top_companies, utils.today_before(729), utils.today(), interval='1h')
    
    combined_df.to_pickle(name)
    

def load_combined_prices(name):
    try:
        df = pd.read_pickle(name)
        return df
    except FileNotFoundError:
        print(f"Error: '{name}' not found. Please run create_pickle() first.")
        return None

def save_data_chunk(X, y, prefix, chunk_dir='./chunks'):
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_id = len(glob.glob(os.path.join(chunk_dir, f'{prefix}data_chunk_*.pkl')))
    with open(os.path.join(chunk_dir, f'{prefix}data_chunk_{chunk_id}.pkl'), 'wb') as f:
        pickle.dump((np.array(X), np.array(y)), f)

def process_symbol_data(symbol, df_with_indicators, seq_length, features, target_function):
    print(f"Processing {symbol}...")
    data = df_with_indicators[[f'{symbol}_Price'] + [f'{symbol}_{feature}' for feature in features]].copy()
    data = target_function(data, symbol)
    
    
    data.dropna(inplace=True)
    
    symbol_X, symbol_y = [], []
    for i in range(len(data) - seq_length):
        symbol_X.append(data[[f'{symbol}_{feature}' for feature in features]].iloc[i:i+seq_length].values)
        symbol_y.append(data[f'{symbol}_Signal'].iloc[i+seq_length])
    
    
    print(f"{symbol} : Preprocessed")
    return symbol_X, symbol_y

from sklearn.preprocessing import OneHotEncoder
def create_and_save_data(symbols, df_with_indicators, seq_length, features, target_function, prefix):
    X, y = [], []
    for symbol in symbols:
        symbol_X, symbol_y = process_symbol_data(symbol, df_with_indicators, seq_length, features, target_function)
        X.extend(symbol_X)
        y.extend(symbol_y)
        
        if len(X) > 4000:
            save_data_chunk(X, y, prefix)
            X, y = [], []
    
    if X:
        save_data_chunk(X, y, prefix)

def load_and_split_data(prefix, chunk_dir='./chunks'):
    X_all, y_all = [], []
    for chunk_file in glob.glob(os.path.join(chunk_dir, f'{prefix}data_chunk_*.pkl')):
        with open(chunk_file, 'rb') as f:
            X_chunk, y_chunk = pickle.load(f)
            X_all.append(X_chunk)
            y_all.append(y_chunk)
    
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    # Check for NaN or infinite values and replace/remove them
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=1e10, neginf=-1e10)

    encoder = OneHotEncoder(sparse=False)
    y_all = encoder.fit_transform(y_all.reshape(-1, 1))
    
    print("Starting train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    print("Split completed")

    return X_train, X_test, y_train, y_test


def TRACK_target_function(data, symbol, lookahead_hours=2, threshold=0.015):
    """
    상승/하락/변동 없음(0.5% 미만 변동)으로 3개의 클래스로 분류합니다.
    """
    data[f'{symbol}_Price_Change'] = data[f'{symbol}_Price'].pct_change(lookahead_hours).fillna(0)

    def classify_change(change):
        if change > threshold:
            return 1  # 상승
        elif change < -threshold:
            return 2  # 하락
        else:
            return 0  # 변동 크지 않음

    data[f'{symbol}_Signal'] = data[f'{symbol}_Price_Change'].apply(classify_change)
    
    return data

from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization, Bidirectional

def create_model(input_shape, loss='categorical_crossentropy'):
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape, 
                           kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
                           kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        BatchNormalization(),
        
        LSTM(32, activation='tanh', return_sequences=True, 
                           kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
                           kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        BatchNormalization(),
        
        LSTM(16, activation='tanh',
             kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
             kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        
        Dense(3, activation='softmax')
    ])
    
    # Optimizer with a lower learning rate
    optimizer = Adam(learning_rate=0.0003, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    return model


def train_model_with_oversampling(model, X_train, y_train):
    # SMOTE를 통해 데이터 오버샘플링
    X_train_resampled, y_train_resampled = oversample_data(X_train, y_train)

    checkpoint = ModelCheckpoint(f'TRACK_univ.h5', monitor='val_loss', save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("Starting TRACK model training with oversampled data...")
    history = model.fit(
        X_train_resampled, y_train_resampled,
        epochs=50,
        batch_size=48,
        validation_split=0.2,
        callbacks=[checkpoint, early_stop]
    )
    return history

def oversample_data(X, y):
    """Applies SMOTE for oversampling."""
    n_samples, timesteps, n_features = X.shape
    X_reshaped = X.reshape(n_samples, timesteps * n_features)  # Flatten the data for SMOTE

    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
    
    # Reshape back to original dimensions
    X_resampled = X_resampled.reshape(-1, timesteps, n_features)
    
    return X_resampled, y_resampled

seq_length = 12
def create_TRACK_model(df_with_indicators, symbols):
    features = track_features
    
    if not glob.glob('./chunks/TRACK_data_chunk_*.pkl'):
        create_and_save_data(symbols, df_with_indicators, seq_length, features, TRACK_target_function, 'TRACK_')
    
    X_train, X_test, y_train, y_test = load_and_split_data('TRACK_')

    # 다중 분류 모델 생성 및 학습
    model = create_model((seq_length, len(features)), loss='categorical_crossentropy')
    history = train_model_with_oversampling(model, X_train, y_train)
    
    model = load_model_with_error_handling('TRACK_univ.h5')
    evaluate_model(model, X_test, y_test)
    
    return model

import ta
from sklearn.preprocessing import MinMaxScaler
track_features = ['EMA_12', 'MACD_diff', 'ATR', 'Bollinger_band_diff', 'RSI', 'Momentum', 'Volume_Change']

def calculate_technical_indicators(df, symbol):
    df[f'{symbol}_EMA_12'] = ta.trend.EMAIndicator(df[symbol+'_Price'], window=5).ema_indicator()
    macd = ta.trend.MACD(df[symbol+'_Price'], window_slow=7, window_fast=3, window_sign=3)
    df[f'{symbol}_MACD_diff'] = macd.macd_diff()
    df[f'{symbol}_ATR'] = ta.volatility.AverageTrueRange(
        high=df[symbol+'_High'], low=df[symbol+'_Low'], close=df[symbol+'_Price'], window=5
    ).average_true_range()
    bollinger = ta.volatility.BollingerBands(df[symbol+'_Price'], window=5)
    df[f'{symbol}_Bollinger_band_diff'] = bollinger.bollinger_hband() - bollinger.bollinger_lband()
    df[f'{symbol}_RSI'] = ta.momentum.RSIIndicator(df[symbol+'_Price'], window=5).rsi()
    df[f'{symbol}_Momentum'] = ta.momentum.ROCIndicator(df[symbol+'_Price'], window=5).roc()
    df[f'{symbol}_Volume_Change'] = df[symbol+'_Volume'].pct_change().fillna(0)
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    scaler = MinMaxScaler()
    for feature in track_features:
        df[f'{symbol}_{feature}'] = scaler.fit_transform(df[[f'{symbol}_{feature}']])
    return df

import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model

# 모델 로드 함수
def load_model_with_error_handling(model_path):
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# 평가 함수
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

if __name__ == "__main__":
    
    #create_pickle(name='1h_sp500_combined_close_volume_prices.pkl')
    combined_prices = load_combined_prices('1h_sp500_combined_close_volume_prices.pkl')
    symbols = set()
    for col in combined_prices.columns:
        symbols.add(col.split('_')[0])
    #symbol_Price, _High, _Low, Volume
    
    if combined_prices is not None:
        df_with_indicators = combined_prices.copy()
        new_indicators = {}
        for stock in symbols:
            temp_df = pd.DataFrame({
                f'{stock}_Price': df_with_indicators[f"{stock}_Price"],
                f'{stock}_High': df_with_indicators[f"{stock}_High"],
                f'{stock}_Low': df_with_indicators[f"{stock}_Low"],
                f'{stock}_Volume' : df_with_indicators[f"{stock}_Volume"]
            })
            temp_df = calculate_technical_indicators(temp_df, stock)
            for indicator in track_features:
                new_indicators[f'{stock}_{indicator}'] = temp_df[f'{stock}_{indicator}']
        df_with_indicators = pd.concat([df_with_indicators, pd.DataFrame(new_indicators)], axis=1)
        create_TRACK_model(df_with_indicators, symbols)