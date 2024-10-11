import os
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.regularizers import l2

import localbns

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
        df = df[['Date', 'Close', 'High', 'Low']]
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

def process_symbol_data(symbol, df_with_indicators, seq_length, features, target_function):
    print(f"Processing {symbol}...")
    data = df_with_indicators[[f'{symbol}'] + [f'{symbol}_{feature}' for feature in features]].copy()
    data = target_function(data, symbol)
    data.dropna(inplace=True)

    symbol_X, symbol_y = [], []
    for i in range(len(data) - seq_length):
        symbol_X.append(data[[f'{symbol}_{feature}' for feature in features]].iloc[i:i+seq_length].values)
        symbol_y.append(data[f'{symbol}_Signal'].iloc[i+seq_length])
    
    # Count the number of 1's and 0's in symbol_y
    ones_count = sum(symbol_y)
    zeros_count = len(symbol_y) - ones_count
    
    print(f"{symbol} : Number of 1's: {ones_count}, Number of 0's: {zeros_count}")
    
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

def create_model(input_shape, loss='binary_crossentropy'):
    model = Sequential([
        LSTM(32, activation='tanh', return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        LSTM(16, activation='tanh', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss=loss, metrics=['accuracy'])
    return model

class BalancedBatchGenerator(Sequence):
    def __init__(self, X, y, batch_size=32):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        
        self.positive_indexes = np.where(y == 1)[0]
        self.negative_indexes = np.where(y == 0)[0]
        self.n_positive = len(self.positive_indexes)
        self.n_negative = len(self.negative_indexes)
        
        # 양성 샘플 수와 음성 샘플 수 중 작은 값을 선택
        self.samples_per_class = min(self.n_positive, self.n_negative, self.batch_size // 2)
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_size = min(self.batch_size, len(self.X) - idx * self.batch_size)
        n_pos = min(self.samples_per_class, batch_size // 2)
        n_neg = batch_size - n_pos
        
        positive_samples = np.random.choice(self.positive_indexes, size=n_pos, replace=True)
        negative_samples = np.random.choice(self.negative_indexes, size=n_neg, replace=True)
        
        batch_indexes = np.concatenate([positive_samples, negative_samples])
        np.random.shuffle(batch_indexes)
        
        return self.X[batch_indexes], self.y[batch_indexes]

def train_model(model, X_train, y_train, model_name):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    checkpoint = ModelCheckpoint(f'LOCALBNS_{model_name}_univ.h5', monitor='val_loss', save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

    print(f"Starting {model_name} model training...")
    
    train_generator = BalancedBatchGenerator(X_train, y_train, batch_size=32)
    val_generator = BalancedBatchGenerator(X_val, y_val, batch_size=32)
    
    history = model.fit(
        train_generator,
        epochs=150,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    return history

def sell_target_function(data, symbol):
    data[f'{symbol}_EMA_Change'] = data[f'{symbol}_EMA_5'].pct_change()
    data[f'{symbol}_ATR_Change'] = data[f'{symbol}_ATR'].pct_change()
    data[f'{symbol}_Signal'] = ((data[f'{symbol}_EMA_Change'].abs() < 0.003) &
                    (data[f'{symbol}_ATR_Change'] > 0.003)).astype(int)
    data.dropna(inplace=True)
    return data

def buy_target_function(data, symbol):
    data['MACD_Change'] = data[f'{symbol}_MACD'].pct_change()
    data['Bollinger_Lower_Change'] = data[f'{symbol}_Bollinger_lband'].pct_change()
    data[f'{symbol}_Signal'] = ((data['MACD_Change'].abs() < 0.001) & 
                      (data['Bollinger_Lower_Change'] < -0.005)).astype(int)
    return data

buy_features = ['MACD', 'Bollinger_lband']
sell_features = ['EMA_5', 'ATR']

def create_buy_model(df_with_indicators, symbols):
    seq_length = localbns.seqlen
    features = buy_features
    
    if not glob.glob('./chunks/buy_data_chunk_*.pkl'):
        create_and_save_data(symbols, df_with_indicators, seq_length, features, buy_target_function, 'buy_')
    
    X_train, X_test, y_train, y_test = load_and_split_data('buy_')
    

    model = create_model((seq_length, len(features)), loss='binary_crossentropy')
    history = train_model(model, X_train, y_train, 'buy')
    
    print("Evaluating buy model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.5f}")
    
    # 예측 분포 출력
    predictions = model.predict(X_test)
    print("Prediction distribution:")
    print(pd.Series(predictions.flatten()).describe())
    
    return model

def create_sell_model(df_with_indicators, symbols):
    seq_length = localbns.seqlen
    features = sell_features
    
    if not glob.glob('./chunks/sell_data_chunk_*.pkl'):
        create_and_save_data(symbols, df_with_indicators, seq_length, features, sell_target_function, 'sell_')
    
    X_train, X_test, y_train, y_test = load_and_split_data('sell_')
    
    model = create_model((seq_length, len(features)), loss='binary_crossentropy')
    history = train_model(model, X_train, y_train, 'sell')
    
    print("Evaluating sell model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.5f}")
    
    # 예측 분포 출력
    predictions = model.predict(X_test)
    print("Prediction distribution:")
    print(pd.Series(predictions.flatten()).describe())
    
    return model

def finetune_model(model, X, y, model_name, epochs=50, batch_size=32):
    checkpoint = ModelCheckpoint(f'best_{model_name}_finetuned_model.h5', monitor='loss', save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    print(f"Fine-tuning {model_name} model...")
    
    train_generator = BalancedBatchGenerator(X, y, batch_size=batch_size)
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=len(train_generator),
        callbacks=[checkpoint, early_stop]
    )
    return model, history

def finetune_buy_model(symbol, df_with_indicators, original_model_path='LOCALBNS_buy_univ.h5'):
    seq_length = localbns.seqlen
    features = buy_features
    symbol_data = df_with_indicators[[f'{symbol}_Price'] + [f'{symbol}_{feature}' for feature in features]].copy()
    symbol_data = buy_target_function(symbol_data, symbol)

    X, y = [], []
    for i in range(len(symbol_data) - seq_length):
        X.append(symbol_data[[f'{symbol}_{feature}' for feature in features]].iloc[i:i+seq_length].values)
        y.append(symbol_data[f'{symbol}_Signal'].iloc[i+seq_length])

    X = np.array(X)
    y = np.array(y)
    
    model = load_model(original_model_path)
    finetuned_model, history = finetune_model(model, X, y, f'buy_{symbol}')
    
    return finetuned_model

def finetune_sell_model(symbol, df_with_indicators, original_model_path='LOCALBNS_sell_univ.h5'):
    seq_length = localbns.seqlen
    features = sell_features
    symbol_data = df_with_indicators[[f'{symbol}_Price'] + [f'{symbol}_{feature}' for feature in features]].copy()
    symbol_data = sell_target_function(symbol_data, symbol)
    
    X, y = [], []
    for i in range(len(symbol_data) - seq_length):
        X.append(symbol_data[[f'{symbol}_{feature}' for feature in features]].iloc[i:i+seq_length].values)
        y.append(symbol_data[f'{symbol}_Signal'].iloc[i+seq_length])
    
    X = np.array(X)
    y = np.array(y)
    
    model = load_model(original_model_path)
    finetuned_model, history = finetune_model(model, X, y, f'sell_{symbol}')
    
    return finetuned_model

if __name__ == "__main__":
    create_pickle()
    combined_prices = load_combined_prices('sp500_combined_close_prices.pkl')
    #combined_prices = localbns.nplog(combined_prices)
    symbols = [col for col in combined_prices.columns if '_' not in col]
    if combined_prices is not None:
        df_with_indicators = combined_prices.copy()
        new_indicators = {}
        for stock in df_with_indicators.columns:
            temp_df = pd.DataFrame({
                f'{stock}_Price': df_with_indicators[stock],
                f'{stock}_High': df_with_indicators[stock],
                f'{stock}_Low': df_with_indicators[stock],
            })
            temp_df = localbns.calculate_technical_indicators(temp_df, stock)
            for indicator in localbns.features:
                new_indicators[f'{stock}_{indicator}'] = temp_df[f'{stock}_{indicator}']
        df_with_indicators = pd.concat([df_with_indicators, pd.DataFrame(new_indicators)], axis=1)
        df_with_indicators.to_pickle('sp500_combined_prices_with_indicators.pkl')
        print("Saved new DataFrame with indicators to 'sp500_combined_prices_with_indicators.pkl'")
        
        buy_model = create_buy_model(df_with_indicators, symbols)
        sell_model = create_sell_model(df_with_indicators, symbols)
        
        '''
        import utils
        raw = utils.load_historical_data("XOM", "2022-01-01", "2024-01-01")
        d = localbns.calculate_technical_indicators(raw, "XOM")
        finetune_buy_model("XOM", d, 'LOCALBNS_buy_univ.h5')
        finetune_sell_model("XOM", d, 'LOCALBNS_sell_univ.h5')
        '''