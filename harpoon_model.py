import utils

from ta import add_all_ta_features
from ta.trend import adx, cci
from ta.volume import on_balance_volume
from ta.momentum import rsi
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
import glob
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from imblearn.combine import SMOTETomek
from collections import Counter
def normalize(df):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

def oversample_data(X, y, min_neighbors=5):
    """Applies SMOTE for oversampling."""
    n_samples, timesteps, n_features = X.shape
    X_reshaped = X.reshape(n_samples, timesteps * n_features)  # Flatten the data for SMOTE

    # k_neighbors를 샘플 수보다 작게 설정
    k_neighbors = min(min_neighbors, n_samples - 1)  # n_samples보다 작은 값으로 설정
    smote = SMOTE(k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
    
    # Reshape back to original dimensions
    X_resampled = X_resampled.reshape(-1, timesteps, n_features)
    
    return X_resampled, y_resampled

features = ['MA20', 'RSI', 'VWAP', 'CMF', 'CCI', 'ADX', 'OBV']
seqlen = 90
window = 90

def target_function(data, symbol):
    data = data.copy()
    target = f'{symbol}_Signal'
    data[target] = 0  # 0: 패턴 없음, 2: V자, 1: 역V자
    
    slope_threshold = 0.01

    for i in range(0, len(data)-window+1):
        window_data = data.iloc[i:i+window]
        
        if len(window_data) < window:
            continue
        cmf = window_data[f"{symbol}_CMF"].values
        vwap = window_data[f"{symbol}_VWAP"].values
        min_point = np.argmin(cmf)
        max_point = np.argmax(cmf)
        if min_point > 0 and min_point < len(cmf) - 1:
            left_slope = (cmf[min_point] - cmf[0]) / (vwap[min_point] - vwap[0] + 1e-6)
            right_slope = (cmf[-1] - cmf[min_point]) / (vwap[-1] - vwap[min_point] + 1e-6)
            if left_slope < -slope_threshold and right_slope > slope_threshold:
                data.loc[data.index[i + min_point], target] = 2 # V
            
            # 역V자 패턴 확인
            elif left_slope > slope_threshold and right_slope < -slope_threshold:
                data.loc[data.index[i + max_point], target] = 1 # Inverted V
            
    return data

def evaluate_model(model, X_test, y_test):
    try:
        print("Evaluating trend model...")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.5f}")
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

    except Exception as e:
        print(f"Error during evaluation: {e}")

def create_model(input_shape, loss='categorical_crossentropy'):
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape),  # LSTM 유닛 수 증가
        Dropout(0.2),
        LSTM(32, activation='tanh', return_sequences=True),
        Dropout(0.2),
        LSTM(16, activation='tanh'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

def train_model_with_oversampling(model, X_train, y_train):
    X_train_resampled, y_train_resampled = oversample_data(X_train, y_train)
    X_train_resampled = X_train_resampled.astype(np.float32)
    y_train_resampled = y_train_resampled.astype(np.float32)  # One-Hot 인코딩된 레이블
    print(f"Starting model training with oversampled data...")
    checkpoint = ModelCheckpoint(f'HARPOON_univ.keras', monitor='val_loss', save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    print(f"Starting HARPOON model training with oversampled data...")
    history = model.fit(
        X_train_resampled, y_train_resampled,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[checkpoint, early_stop]
    )
    return history

def create_harpoon(df_with_indicators, symbols):
    if not glob.glob('./chunks/harpoon_data_chunk_*.pkl'):
        utils.create_and_save_data_012(symbols, df_with_indicators, seqlen, features, target_function, 'harpoon_')
    X_train, X_test, y_train, y_test = utils.load_and_split_data_onehot('harpoon_')
    model = create_model((seqlen, len(features)), loss='categorical_crossentropy')
    history = train_model_with_oversampling(model, X_train, y_train)
    print("Evaluating trend model...")
    model = load_model('HARPOON_univ.keras')
    evaluate_model(model, X_test, y_test)

def calculate_technical_indicators(df, symbol):
    df = df.copy()
    df[f'{symbol}_MA20'] = df[f'{symbol}_Close'].rolling(window=20).mean()
    df[f'{symbol}_MA50'] = df[f'{symbol}_Close'].rolling(window=50).mean()
    df[f'{symbol}_RSI'] = rsi(df[f'{symbol}_Close'], window=14)
    df[f'{symbol}_VWAP'] = (df[f'{symbol}_Close'] * df[f'{symbol}_Volume']).cumsum() / df[f'{symbol}_Volume'].cumsum()
    df[f'{symbol}_CMF'] = ((df[f'{symbol}_Close'] - df[f'{symbol}_Low']) - (df[f'{symbol}_High'] - df[f'{symbol}_Close'])) / \
                   (df[f'{symbol}_High'] - df[f'{symbol}_Low']) * df[f'{symbol}_Volume']
    df[f'{symbol}_CMF'] = df[f'{symbol}_CMF'].rolling(window=20).mean()
    df[f'{symbol}_CCI'] = cci(df[f'{symbol}_High'], df[f'{symbol}_Low'], df[f'{symbol}_Close'], window=20)
    df[f'{symbol}_ADX'] = adx(df[f'{symbol}_High'], df[f'{symbol}_Low'], df[f'{symbol}_Close'], window=14)
    df[f'{symbol}_OBV'] = on_balance_volume(df[f'{symbol}_Close'], df[f'{symbol}_Volume'])

    df.dropna(inplace=True)

    scaler = StandardScaler()
    for feature in features:
        df[f'{symbol}_{feature}'] = scaler.fit_transform(df[[f'{symbol}_{feature}']])

    return df

def _finetune_model(model, X, y, model_name, epochs=15, batch_size=64):
    checkpoint = ModelCheckpoint(f'best_{model_name}_finetuned_model.keras', monitor='loss', save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
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

def finetune_model(symbol, df_with_indicators, original_model_path='HARPOON_univ.keras'):
    symbol_data = df_with_indicators[[f'{symbol}_{feature}' for feature in features]].copy()
    symbol_data = target_function(symbol_data, symbol)
    
    X, y = [], []
    for i in range(len(symbol_data) - seqlen):
        X.append(symbol_data[[f'{symbol}_{feature}' for feature in features]].iloc[i:i+seqlen].values)
        y.append(symbol_data[f'{symbol}_Signal'].iloc[i+seqlen])
    
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

    finetuned_model, history = _finetune_model(model, X_resampled, y_resampled, f'HARPOON_{symbol}')
    
    return finetuned_model

def predict(model, raw, symbol):
    df = calculate_technical_indicators(raw, symbol)

    x = df[[f"{symbol}_{feature}" for feature in features]].values

    x = np.expand_dims(x, axis=0)
    return model.predict(x, verbose=0)[0]

if __name__ == "__main__":
    combined_prices = utils.load_combined_prices('sp500_combined_close_prices.pkl')
    if combined_prices is not None:
        symbols = {col.split('_')[0] for col in combined_prices.columns}
        df_with_indicators = combined_prices.copy()
        new_indicators = {}
        for stock in symbols:
            temp_df = pd.DataFrame({
                f'{stock}_Open': combined_prices[f"{stock}_Open"],
                f'{stock}_High': combined_prices[f"{stock}_High"],
                f'{stock}_Low': combined_prices[f"{stock}_Low"],
                f'{stock}_Close': combined_prices[f"{stock}_Close"],
                f'{stock}_Volume': combined_prices[f"{stock}_Volume"]
            })
            df = calculate_technical_indicators(temp_df, stock)
            for indicator in features:
                new_indicators[f'{stock}_{indicator}'] = temp_df[f'{stock}_{indicator}']
        df_with_indicators = pd.concat([df_with_indicators, pd.DataFrame(new_indicators)], axis=1)
            
        create_harpoon(df_with_indicators, symbols)