import os
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle

import localbns

def create_pickle(directory='./stock_market_data/sp500/', name = 'sp500_combined_close_prices.pkl'):
    csv_files = glob.glob(os.path.join(directory, 'csv/*.csv'))
    combined_df = pd.DataFrame()
    # List of top 25 S&P 500 companies by market cap
    top_companies = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'GOOG', 'TSLA', 'BRK.B',
        'UNH', 'JPM', 'JNJ', 'V', 'XOM', 'PG', 'MA', 'LLY', 'HD', 'AVGO', 'CVX',
        'ABBV', 'MRK', 'PEP', 'KO', 'BAC'
    ]

    # Filter CSV files to include only the top companies
    csv_files = [f for f in csv_files if any(company in f for company in top_companies)]
    print(f"Number of top companies found: {len(csv_files)}")
    for file in csv_files:
        df = pd.read_csv(file)
        stock_name = os.path.basename(file).split('.')[0]
        df = df[['Date', 'Open', 'Close','Volume', 'High', 'Low']]
        df['Open'] = df['Open'].astype(float)
        df['Close'] = df['Close'].astype(float)
        df['Volume'] = df['Volume'].astype(float)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')  # Convert date format
        df = df.rename(columns={'Open': f"{stock_name}_Open", 'Close': f"{stock_name}_Close", 'Volume': f"{stock_name}_Volume", 'High' : f"{stock_name}_High", 'Low' : f"{stock_name}_Low"})
        df.set_index('Date', inplace=True)
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = combined_df.join(df, how='outer')
    combined_df.sort_index(inplace=True)
    # Filter data from 2005 onwards
    combined_df = combined_df.loc['2010-01-01':]
    original_columns = len(combined_df.columns)
    # Remove columns with 30 or more consecutive NaN values
    combined_df = combined_df.dropna(axis=1, thresh=len(combined_df) - 29)
    # Remove columns where all values are NaN
    combined_df = combined_df.dropna(axis=1, how='all')
    removed_columns = original_columns - len(combined_df.columns)
    print(f"Filtered data from 2010 onwards.")
    print(f"Removed {removed_columns} columns where all values were NaN.")
    print(f"Remaining columns: {len(combined_df.columns)}")
    combined_df.dropna(inplace=True)
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
    print(f"{symbol} : Number of 1's: {ones_count}, Number of 0's: {zeros_count}, Number of 2's: {twos_count}")
    
    return symbol_X, symbol_y

from sklearn.preprocessing import OneHotEncoder
def create_and_save_data(symbols, df_with_indicators, seq_length, features, target_function, prefix):
    X, y = [], []
    for symbol in symbols:
        symbol_X, symbol_y = process_symbol_data(symbol, df_with_indicators, seq_length, features, target_function)
        X.extend(symbol_X)
        y.extend(symbol_y)
        
        if len(X) > 4000:  # Adjust this threshold as needed
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
    y_all = encoder.fit_transform(np.array(y_all).reshape(-1, 1))

    print("Starting train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    print("Split completed")

    return X_train, X_test, y_train, y_test

# 모델 생성 함수
def create_model(input_shape, loss='mse'):
    model = Sequential([
        LSTM(128, activation='tanh', return_sequences=True, input_shape=input_shape),  # LSTM 유닛 수 증가
        Dropout(0.4),
        LSTM(64, activation='tanh', return_sequences=True),
        Dropout(0.4),
        LSTM(32, activation='tanh'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# GAP 타겟 함수
def GAP_target_function(data, symbol, lookahead_days=1):
    data[f'{symbol}_Signal'] = data[f'{symbol}_Gap_Size'].apply(
        lambda x: 1 if x >= 0.05 else (0 if x <= -0.05 else 2)
    )
    return data

import numpy as np
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.callbacks import ModelCheckpoint, EarlyStopping
import smote_variants as sv

import tensorflow as tf

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        # Convert y_true and y_pred to float32
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        # Calculate the focal loss
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)  # Adjust alpha for each class
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)  # Probability for true class

        # Focal loss formula
        fl = -alpha_t * tf.pow((1 - p_t), gamma) * tf.math.log(p_t + 1e-8)
        
        return tf.reduce_mean(tf.reduce_sum(fl, axis=1))  # Mean loss across all samples
    
    return focal_loss_fixed

from imblearn.over_sampling import SMOTE

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
    
    # Calculate class weights for balancing
    y_train_1d = np.argmax(y_train, axis=1)  # Assuming y_train_resampled is one-hot encoded

    # Calculate class weights for balancing
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train_1d),  # Use 1D class labels
        y=y_train_1d
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"Starting model training with oversampled data...")
    
    checkpoint = ModelCheckpoint(f'GAP_univ.h5', monitor='val_loss', save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print(f"Starting gap model training with oversampled data...")
    history = model.fit(
        X_train_resampled, y_train_resampled,
        epochs=50,
        batch_size=48,
        validation_split=0.2,
        callbacks=[checkpoint, early_stop],
        #class_weight=class_weight_dict
    )
    return history

import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow import keras

# 모델 로드 함수
def load_model_with_error_handling(model_path):
    
    return keras.models.load_model(model_path)
    
    with keras.utils.custom_object_scope({'focal_loss_fixed': focal_loss()}):
        loaded_model = keras.models.load_model(model_path)
        return loaded_model

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

# 학습 데이터 준비 함수 (기존 틀 유지)
seq_length = 60
def create_GAP_model(df_with_indicators, symbols):
    features = gap_features
    
    if not glob.glob('./chunks/gap_data_chunk_*.pkl'):
        create_and_save_data(symbols, df_with_indicators, seq_length, features, GAP_target_function, 'gap_')
    
    X_train, X_test, y_train, y_test = load_and_split_data('gap_')

    model = create_model((seq_length, len(features)), loss='categorical_crossentropy')
    history = train_model_with_oversampling(model, X_train, y_train)
    print("Evaluating trend model...")
    
    model = load_model_with_error_handling('GAP_univ.h5')
    evaluate_model(model, X_test, y_test)
    
    return model


import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

gap_features = ['EMA_12', 'RSI', 'ATR', 'Bollinger_band_diff', 'Volume_Change', 'Gap_Size']
gap_features_normalized = ['EMA_12', 'RSI', 'ATR', 'Bollinger_band_diff', 'Volume_Change']
def calculate_technical_indicators(df, symbol):
    df[f'{symbol}_EMA_12'] = ta.trend.EMAIndicator(df[symbol+'_Open'], window=12).ema_indicator()
    df[f'{symbol}_RSI'] = ta.momentum.RSIIndicator(df[symbol+'_Open'], window=14).rsi()
    df[f'{symbol}_ATR'] = ta.volatility.AverageTrueRange(df[symbol+'_High'], df[symbol+'_Low'], df[symbol+'_Open'], window=14).average_true_range()
    df[f'{symbol}_Bollinger_hband'] = ta.volatility.BollingerBands(df[symbol+'_Open']).bollinger_hband()
    df[f'{symbol}_Bollinger_lband'] = ta.volatility.BollingerBands(df[symbol+'_Open']).bollinger_lband()
    df[f'{symbol}_Bollinger_band_diff'] = df[f'{symbol}_Bollinger_hband'] - df[f'{symbol}_Bollinger_lband']
    df[f'{symbol}_Volume_Change'] = df[symbol+'_Volume'].pct_change().fillna(0)
    df[f'{symbol}_Gap_Size'] = (df[symbol+'_Close'] - df[symbol+'_Open']) / df[symbol+'_Open']
    df.dropna(inplace=True)
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    scaler = StandardScaler()
    for feature in gap_features_normalized:
        df[f'{symbol}_{feature}'] = scaler.fit_transform(df[[f'{symbol}_{feature}']])
    
    
    return df

if __name__ == "__main__":
    create_pickle(name='sp500_combined_close_volume_prices.pkl')
    combined_prices = load_combined_prices('sp500_combined_close_volume_prices.pkl')
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
            for indicator in gap_features:
                new_indicators[f'{stock}_{indicator}'] = temp_df[f'{stock}_{indicator}']
        df_with_indicators = pd.concat([df_with_indicators, pd.DataFrame(new_indicators)], axis=1)
        df_with_indicators.to_pickle('sp500_combined_prices_with_indicators_GAP.pkl')
        print("Saved new DataFrame with indicators to 'sp500_combined_prices_with_indicators_GAP.pkl'")
        create_GAP_model(df_with_indicators, symbols)