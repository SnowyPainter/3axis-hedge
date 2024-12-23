import sys, os
sys.path.append('../')

import os
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import pickle
from joblib import dump, load
import ta
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def create_pickle(directory='../stock_market_data/sp500/', name='sp500_combined_close_prices.pkl'):
    csv_files = glob.glob(os.path.join(directory, 'csv/*.csv'))
    combined_df = pd.DataFrame()
    top_companies = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'GOOG', 'TSLA', 'BRK.B',
        'UNH', 'JPM', 'JNJ', 'V', 'XOM', 'PG', 'MA', 'LLY', 'HD', 'AVGO', 'CVX',
        'ABBV', 'MRK', 'PEP', 'KO', 'BAC'
    ]

    csv_files = [f for f in csv_files if any(company in f for company in top_companies)]
    print(f"Number of top companies found: {len(csv_files)}")
    
    for file in csv_files:
        df = pd.read_csv(file)
        stock_name = os.path.basename(file).split('.')[0]
        df = df[['Date', 'Open', 'Close', 'Volume', 'High', 'Low']]
        df['Open'] = df['Open'].astype(float)
        df['Close'] = df['Close'].astype(float)
        df['Volume'] = df['Volume'].astype(float)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        df = df.rename(columns={
            'Open': f"{stock_name}_Open", 
            'Close': f"{stock_name}_Close", 
            'Volume': f"{stock_name}_Volume", 
            'High': f"{stock_name}_High", 
            'Low': f"{stock_name}_Low"
        })
        df.set_index('Date', inplace=True)
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = combined_df.join(df, how='outer')
            
    combined_df.sort_index(inplace=True)
    combined_df = combined_df.loc['2010-01-01':]
    combined_df = combined_df.dropna(axis=1, thresh=len(combined_df) - 29)
    combined_df = combined_df.dropna(axis=1, how='all')
    combined_df.dropna(inplace=True)
    combined_df.to_pickle(name)

def load_combined_prices(name):
    try:
        return pd.read_pickle(name)
    except FileNotFoundError:
        print(f"Error: '{name}' not found. Please run create_pickle() first.")
        return None

features = [
    'EMA_12', 'RSI', 'ATR', 'Bollinger_band_diff', 'Volume_Change', 'Gap_Size'
]

features_normalized = [
    'EMA_12', 'RSI', 'ATR', 'Bollinger_band_diff', 'Volume_Change'
]

def calculate_technical_indicators(df, symbol):
    # Calculate all technical indicators
    df[f'{symbol}_EMA_12'] = ta.trend.EMAIndicator(df[symbol+'_Open'], window=12).ema_indicator()
    df[f'{symbol}_RSI'] = ta.momentum.RSIIndicator(df[symbol+'_Open'], window=14).rsi()
    df[f'{symbol}_ATR'] = ta.volatility.AverageTrueRange(df[symbol+'_High'], df[symbol+'_Low'], df[symbol+'_Open'], window=14).average_true_range()
    df[f'{symbol}_Bollinger_hband'] = ta.volatility.BollingerBands(df[symbol+'_Open']).bollinger_hband()
    df[f'{symbol}_Bollinger_lband'] = ta.volatility.BollingerBands(df[symbol+'_Open']).bollinger_lband()
    df[f'{symbol}_Bollinger_band_diff'] = df[f'{symbol}_Bollinger_hband'] - df[f'{symbol}_Bollinger_lband']
    df[f'{symbol}_Volume_Change'] = df[symbol+'_Volume'].pct_change().fillna(0)
    df[f'{symbol}_Gap_Size'] = (df[symbol+'_Close'] - df[symbol+'_Open']) / df[symbol+'_Open']
    macd = ta.trend.MACD(df[symbol+'_Open'])
    df[f'{symbol}_MACD'] = macd.macd()
    df[f'{symbol}_Stoch'] = ta.momentum.StochasticOscillator(df[symbol+'_High'], df[symbol+'_Low'], df[symbol+'_Open']).stoch()
    df[f'{symbol}_WilliamsR'] = ta.momentum.WilliamsRIndicator(df[symbol+'_High'], df[symbol+'_Low'], df[symbol+'_Open']).williams_r()
    df[f'{symbol}_Price_ROC'] = df[symbol+'_Open'].pct_change(periods=14)
    df[f'{symbol}_CCI'] = ta.trend.CCIIndicator(df[symbol+'_High'], df[symbol+'_Low'], df[symbol+'_Open'], window=20).cci()
    df[f'{symbol}_ADX'] = ta.trend.ADXIndicator(df[symbol+'_High'], df[symbol+'_Low'], df[symbol+'_Open']).adx()
    df[f'{symbol}_Momentum'] = df[symbol+'_Open'].diff(periods=10)
    df[f'{symbol}_DMI'] = ta.trend.ADXIndicator(df[symbol+'_High'], df[symbol+'_Low'], df[symbol+'_Open']).adx_pos()
    df[f'{symbol}_VWAP'] = (df[symbol+'_Volume'] * df[symbol+'_Open']).cumsum() / df[symbol+'_Volume'].cumsum()
    
    # Clean up data
    df.dropna(inplace=True)
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Normalize features
    scaler = StandardScaler()
    for feature in features_normalized:
        df[f'{symbol}_{feature}'] = scaler.fit_transform(df[[f'{symbol}_{feature}']])

    return df

def create_features_and_labels(df_with_indicators, symbol, seq_length):
    print(f"Processing {symbol}...")
    
    # Prepare features
    feature_columns = [f'{symbol}_{feature}' for feature in features_normalized]

    X = df_with_indicators[feature_columns].values
    
    # Create labels (1: up, 0: down, 2: neutral)
    df_with_indicators[f'{symbol}_Signal'] = df_with_indicators[f'{symbol}_Gap_Size'].apply(
        lambda x: 1 if x >= 0.01 else (0 if x <= -0.01 else 2)
    )
    y = df_with_indicators[f'{symbol}_Signal'].values
    
    # Create sequences
    X_sequences, y_sequences = [], []
    for i in range(len(X) - seq_length):
        X_sequences.append(X[i:i+seq_length])
        y_sequences.append(y[i+seq_length])
    
    return np.array(X_sequences), np.array(y_sequences)

def train_random_forest_model(X_train, y_train):
    # Reshape the 3D sequence data to 2D for Random Forest
    n_samples, seq_length, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(n_samples, seq_length * n_features)
    
    # Create and train the Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    print("Training Random Forest model...")
    rf_model.fit(X_train_reshaped, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test):
    # Reshape test data
    n_samples, seq_length, n_features = X_test.shape
    X_test_reshaped = X_test.reshape(n_samples, seq_length * n_features)
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test_reshaped)
    original_accuracy = accuracy_score(y_test, y_pred)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    import features
    features.features_MDA_importances(X_test_reshaped, y_test, model, original_accuracy, features_normalized)

if __name__ == "__main__":
    # Parameters
    seq_length = 30
    
    # Create or load data
    create_pickle(name='sp500_combined_close_volume_prices.pkl')
    combined_prices = load_combined_prices('sp500_combined_close_volume_prices.pkl')
    
    if combined_prices is not None:
        # Get unique symbols
        symbols = list(set(col.split('_')[0] for col in combined_prices.columns))
        df_with_indicators = combined_prices.copy()
        new_indicators = {}
        for symbol in symbols:
            temp_df = pd.DataFrame({
                f'{symbol}_Open': df_with_indicators[f"{symbol}_Open"],
                f'{symbol}_Close': df_with_indicators[f"{symbol}_Close"],
                f'{symbol}_High': df_with_indicators[f"{symbol}_High"],
                f'{symbol}_Low': df_with_indicators[f"{symbol}_Low"],
                f'{symbol}_Volume': df_with_indicators[f"{symbol}_Volume"]
            })
            temp_df = calculate_technical_indicators(temp_df, symbol)
            for indicator in features:
                new_indicators[f'{symbol}_{indicator}'] = temp_df[f'{symbol}_{indicator}']
        
        df_with_indicators = pd.concat([df_with_indicators, pd.DataFrame(new_indicators)], axis=1)
        df_with_indicators.dropna(inplace=True)
        df_with_indicators.to_pickle('./gap-light-df-indicator-30.pkl')
        X_all, y_all = [], []
        for symbol in symbols:
            X_symbol, y_symbol = create_features_and_labels(df_with_indicators, symbol, seq_length)
            X_all.append(X_symbol)
            y_all.append(y_symbol)
        
        # Combine all data
        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_all.reshape(len(X_all), -1), y_all)
        X_resampled = X_resampled.reshape(-1, seq_length, X_all.shape[2])  # 원래 형태 복원

        print(X_resampled.shape, y_resampled.shape)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42
        )
        
        # Train and evaluate model
        model = train_random_forest_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        
        # Save the model
        dump(model, 'gap-light.joblib')
        print("\nModel saved as 'gap-light.joblib'")