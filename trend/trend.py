import os
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle

import localbns.localbns as localbns

def create_pickle(directory='./stock_market_data/sp500/', pickle = 'sp500_combined_close_prices.pkl'):
    csv_files = glob.glob(os.path.join(directory, 'csv/*.csv'))
    combined_df = pd.DataFrame()
    # List of top 25 S&P 500 companies by market cap
    top_companies = [
        'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'BRK-B', 'AVGO', 'GOOG', 'LLY',
        'TSLA', 'JPM', 'UNH', 'XOM', 'V', 'MA', 'PG', 'COST', 'JNJ', 'HD',
        'WMT', 'ABBV', 'NFLX', 'MRK', 'KO'
    ]

    # Filter CSV files to include only the top companies
    csv_files = [f for f in csv_files if any(company in f for company in top_companies)]
    print(f"Number of top companies found: {len(csv_files)}")
    for file in csv_files:
        df = pd.read_csv(file)
        stock_name = os.path.basename(file).split('.')[0]
        df = df[['Date', 'Close', 'Volume', 'High', 'Low']]
        df['Close'] = df['Close'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')  # Convert date format
        df = df.rename(columns={'Close': stock_name, 'Volume': f"{stock_name}_Volume", 'High' : f"{stock_name}_High", 'Low' : f"{stock_name}_Low"})
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
    combined_df.to_pickle(pickle)

def load_combined_prices(pickle):
    try:
        df = pd.read_pickle(pickle)
        return df
    except FileNotFoundError:
        print(f"Error: '{pickle}' not found. Please run create_pickle() first.")
        return None

def save_data_chunk(X, y, prefix, chunk_dir='./chunks'):
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_id = len(glob.glob(os.path.join(chunk_dir, f'{prefix}data_chunk_*.pkl')))
    with open(os.path.join(chunk_dir, f'{prefix}data_chunk_{chunk_id}.pkl'), 'wb') as f:
        pickle.dump((np.array(X), np.array(y)), f)

def process_symbol_data(symbol, df_with_indicators, seq_length, features, target_function):
    print(f"Processing {symbol}...")
    data = df_with_indicators[[f'{symbol}'] + [f'{symbol}_{feature}' for feature in features]].copy()
    data = localbns.normalize(data)
    
    data = target_function(data, symbol)
    data.dropna(inplace=True)
    
    symbol_X, symbol_y = [], []
    for i in range(len(data) - seq_length):
        symbol_X.append(data[[f'{symbol}_{feature}' for feature in features]].iloc[i:i+seq_length].values)
        symbol_y.append(data[f'{symbol}_Signal'].iloc[i+seq_length])
    
    print(f"{symbol} : Preprocessed")
    return symbol_X, symbol_y

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
    
    print("Starting train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    print("Split completed")
    return X_train, X_test, y_train, y_test

# 모델 생성 함수
def create_model(input_shape, loss='mse'):
    model = Sequential([
        LSTM(128, activation='tanh', return_sequences=True, input_shape=input_shape),  # LSTM 유닛 증가
        Dropout(0.4),
        LSTM(64, activation='tanh', return_sequences=True),
        Dropout(0.4),
        LSTM(32, activation='tanh'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0003), loss=loss, metrics=['accuracy'])
    return model

# 학습 함수
def train_model(model, X_train, y_train):
    checkpoint = ModelCheckpoint(f'TREND_univ.h5', monitor='val_loss', save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print(f"Starting trend model training...")
    history = model.fit(
        X_train, y_train, 
        epochs=50, 
        batch_size=24, 
        validation_split=0.2, 
        callbacks=[checkpoint, early_stop]
    )
    return history

# 신호 계산 함수 (며칠 내에 오르면 신호로 처리)
def trend_target_function(data, symbol, lookahead_days=7):
    data[f'{symbol}_Price_Change'] = data[f'{symbol}'].pct_change(lookahead_days)
    data[f'{symbol}_Signal'] = (data[f'{symbol}_Price_Change'] > 0).astype(int)
    return data

# 학습 데이터 준비 함수 (기존 틀 유지)
seq_length = 60
def create_TREND_model(df_with_indicators, symbols):
    features = ['EMA_12', 'RSI', 'MACD', 'Bollinger_band_diff', 'ATR']
    
    if not glob.glob('./chunks/trend_data_chunk_*.pkl'):
        create_and_save_data(symbols, df_with_indicators, seq_length, features, trend_target_function, 'trend_')
    
    X_train, X_test, y_train, y_test = load_and_split_data('trend_')
    
    # 모델 생성 및 학습
    model = create_model((seq_length, len(features)), loss='binary_crossentropy')
    history = train_model(model, X_train, y_train)
    
    # 모델 평가
    print("Evaluating trend model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.5f}")
    
    return model

import ta
features = ['SMA_12', 'EMA_12', 'RSI', 'MACD', 'Bollinger_hband', 'Bollinger_lband', 'ATR', 'Bollinger_band_diff']
def calculate_technical_indicators(df, symbol):
    df[f'{symbol}_SMA_12'] = ta.trend.SMAIndicator(df[symbol+'_Price'], window=12).sma_indicator()
    df[f'{symbol}_EMA_12'] = ta.trend.EMAIndicator(df[symbol+'_Price'], window=12).ema_indicator()
    df[f'{symbol}_RSI'] = ta.momentum.RSIIndicator(df[symbol+'_Price'], window=24).rsi()
    df[f'{symbol}_MACD'] = ta.trend.MACD(df[symbol+'_Price']).macd()
    df[f'{symbol}_Bollinger_hband'] = ta.volatility.BollingerBands(df[symbol+'_Price']).bollinger_hband()
    df[f'{symbol}_Bollinger_lband'] = ta.volatility.BollingerBands(df[symbol+'_Price']).bollinger_lband()
    df[f'{symbol}_Bollinger_band_diff'] = df[f'{symbol}_Bollinger_hband'] - df[f'{symbol}_Bollinger_lband']
    df[f'{symbol}_ATR'] = ta.volatility.AverageTrueRange(df[symbol+'_High'], df[symbol+'_Low'], df[symbol+'_Price'], window=14).average_true_range()
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    create_pickle()
    
    combined_prices = load_combined_prices('sp500_combined_close_prices.pkl')
    import utils
    market = utils.load_historical_data("^IXIC", "2010-01-01", "2024-01-01", interval='1d')
    combined_prices.index = pd.to_datetime(combined_prices.index)
    market.index = pd.to_datetime(market.index)
    combined_data = pd.merge(combined_prices, market, left_index=True, right_index=True, how='inner')
    combined_data = combined_data.dropna()
    combined_prices = combined_data
    
    symbols = [col for col in combined_prices.columns if '_' not in col]
    if combined_prices is not None:
        df_with_indicators = combined_prices.copy()
        new_indicators = {}
        for stock in symbols:
            temp_df = pd.DataFrame({
                f'{stock}_Price': df_with_indicators[stock],
                f'{stock}_High': df_with_indicators[f"{stock}_High"],
                f'{stock}_Low': df_with_indicators[f"{stock}_Low"],
                f'{stock}_Volume' : df_with_indicators[f"{stock}_Volume"]
            })
            temp_df = calculate_technical_indicators(temp_df, stock)
            for indicator in features:
                new_indicators[f'{stock}_{indicator}'] = temp_df[f'{stock}_{indicator}']
        df_with_indicators = pd.concat([df_with_indicators, pd.DataFrame(new_indicators)], axis=1)
        df_with_indicators.to_pickle('sp500_combined_prices_with_indicators_TREND.pkl')
        print("Saved new DataFrame with indicators to 'sp500_combined_prices_with_indicators_TREND.pkl'")
        
        create_TREND_model(df_with_indicators, symbols)