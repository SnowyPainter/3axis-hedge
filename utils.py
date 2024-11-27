import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from collections import namedtuple
import pytz
import os, re
import pickle
import glob
from ta import add_all_ta_features
from ta.trend import adx, cci
from ta.volume import on_balance_volume
from ta.momentum import rsi
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

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
        df = df[['Date', 'Open', 'Close', 'High', 'Low', 'Volume']]
        df['Open'] = df['Open'].astype(float)
        df['Close'] = df['Close'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
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
    combined_df.to_pickle(pickle)

def load_combined_prices(pickle):
    try:
        df = pd.read_pickle(pickle)
        print(f"Successfully loaded combined close prices from {pickle}")
        print(f"\nShape of the DataFrame: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: '{pickle}' not found. Please run create_pickle() first.")
        return None

def save_data_chunk(X, y, prefix, chunk_dir='./chunks'):
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_id = len(glob.glob(os.path.join(chunk_dir, f'{prefix}data_chunk_*.pkl')))
    with open(os.path.join(chunk_dir, f'{prefix}data_chunk_{chunk_id}.pkl'), 'wb') as f:
        pickle.dump((np.array(X), np.array(y)), f)

def load_and_split_data_onehot(prefix, chunk_dir='./chunks'):
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

    encoder = OneHotEncoder(sparse_output=False)
    y_all = encoder.fit_transform(np.array(y_all).reshape(-1, 1))

    print("Starting train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    print("Split completed")

    return X_train, X_test, y_train, y_test

def today(tz = 'Asia/Seoul'):
        return datetime.now(pytz.timezone(tz))
def today_before(day, tz = 'Asia/Seoul'):
    return datetime.now(pytz.timezone(tz)) - timedelta(days=day)

def get_OHLCV(symbol, start, end, interval='1d'):
    d = yf.download(symbol, start=start, end=end, interval=interval)
    d.index = pd.to_datetime(d.index, format="%Y-%m-%d %H:%M:%S%z")
    return d

def process_symbol_data_012(symbol, df_with_indicators, seq_length, features, target_function):
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

def create_and_save_data_012(symbols, df_with_indicators, seq_length, features, target_function, prefix):
    X, y = [], []
    for symbol in symbols:
        symbol_X, symbol_y = process_symbol_data_012(symbol, df_with_indicators, seq_length, features, target_function)
        X.extend(symbol_X)
        y.extend(symbol_y)
        
        if len(X) > 4000:  # Adjust this threshold as needed
            save_data_chunk(X, y, prefix)
            X, y = [], []
    
    if X:
        save_data_chunk(X, y, prefix)

def tech(ohlcv):
    ohlcv = ohlcv.copy()
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

    ohlcv.dropna(inplace=True)

    # Min-Max 스케일링 적용
    scaler = MinMaxScaler()
    scaled_features = ['MA20', 'MA50', 'RSI', 'VWAP', 'CMF', 'CCI', 'ADX', 'OBV']
    ohlcv[scaled_features] = scaler.fit_transform(ohlcv[scaled_features])

    return ohlcv

def load_historical_data(symbol, start, end, interval='1d'):
    d = yf.download(symbol, start=start, end=end, interval=interval)
    d.rename(columns={'Close': symbol+'_Price', 'Volume' : symbol+"_Volume", 'High' : symbol+"_High", 'Low' : symbol+"_Low"}, inplace=True)
    d.index = pd.to_datetime(d.index, format="%Y-%m-%d %H:%M:%S%z")
    return d[[symbol+'_Price', symbol+"_Volume", symbol+"_High", symbol+"_Low"]]

def merge_dfs(dfs):
    if len(dfs) == 0:
        return pd.DataFrame({})
    merged = dfs[0]
    for i in range(1, len(dfs)):
        merged = merged.join(dfs[i])
    return merged

def load_historical_datas(symbols, start, end, interval='1d'):
    dfs = []
    edit_symbols = symbols.copy()
    for symbol in symbols:
        df = load_historical_data(symbol, start, end, interval)
        if df.empty == True:
            edit_symbols.remove(symbol)
            continue
        dfs.append(df)
    
    merged = merge_dfs(dfs)
    merged.dropna(inplace=True)
    
    return merged, edit_symbols

def models_exists(symbols):
    flag = True
    for symbol in symbols:
        flag = os.path.exists(f"{symbol}_Buy.keras")
        flag = os.path.exists(f"{symbol}_Sell.keras")
    return flag