import sys, os
sys.path.append('../')
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, LayerNormalization, Add, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from tensorflow.keras.mixed_precision import Policy, set_global_policy
#policy = Policy('mixed_float16')
#set_global_policy(policy)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

features = [
    'RSI', 'ATR', 'Bollinger_band_diff', 'Volume_Change', 'MACD', 
    'Stoch', 'WilliamsR', 'ADX', 'Momentum', 'Gap_Size'
]

features_normalized = [
    'RSI', 'ATR', 'Bollinger_band_diff', 'Volume_Change', 'MACD',
    'Stoch', 'WilliamsR', 'ADX', 'Momentum'
]

def create_features_and_labels(df_with_indicators, symbol, seq_length):
    print(f"Processing {symbol}...")
    feature_columns = [f'{symbol}_{feature}' for feature in features_normalized]
    X = df_with_indicators[feature_columns].values
    df_with_indicators[f'{symbol}_Signal'] = df_with_indicators[f'{symbol}_Gap_Size'].apply(
        lambda x: 1 if x >= 0.02 else (0 if x <= -0.02 else 2)
    )
    y = df_with_indicators[f'{symbol}_Signal'].values
    X_sequences, y_sequences = [], []
    for i in range(len(X) - seq_length):
        X_sequences.append(X[i:i+seq_length])
        y_sequences.append(y[i+seq_length])
    
    return np.array(X_sequences), np.array(y_sequences)

def create_lstm_model(seq_length, n_features, n_classes=3):
    inputs = Input(shape=(seq_length, n_features))
    lstm1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(inputs)
    norm1 = LayerNormalization()(lstm1)
    lstm2 = Bidirectional(LSTM(64))(norm1)
    norm2 = LayerNormalization()(lstm2)
    dense1 = Dense(128, activation='relu')(norm2)
    drop1 = Dropout(0.2)(dense1)
    dense2 = Dense(64, activation='relu')(drop1)
    
    output = Dense(n_classes, activation='softmax')(dense2)
    
    model = Model(inputs=inputs, outputs=output)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0005,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(X_train, y_train, X_val, y_val, seq_length, n_features, model_name='gap-d1.h5'):
    model = create_lstm_model(seq_length, n_features)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_name,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Model training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=256,
        callbacks=callbacks,
        verbose=1,
        workers=4,
        use_multiprocessing=True
    )
    
    return model, history


def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_classes))

import ta
from sklearn.preprocessing import StandardScaler

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

def create_df_indicators(combined_prices):
    if combined_prices is not None:
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
        df_with_indicators.to_pickle('./gap-d1-df-indicator.pkl')
        return symbols, df_with_indicators

if __name__ == "__main__":
    seq_length = 60
    combined_prices = pd.read_pickle('sp500_combined_close_volume_prices.pkl')
    symbols = list(set(col.split('_')[0] for col in combined_prices.columns))
    
    # create_df_indicators 하고 나서 이거 하면 Killed 됨.

    df_with_indicators = pd.read_pickle('./gap-d1-df-indicator.pkl')
    
    X_all, y_all = [], []
    for symbol in symbols:
        X_symbol, y_symbol = create_features_and_labels(df_with_indicators, symbol, seq_length)
        X_all.append(X_symbol)
        y_all.append(y_symbol)
    
    del df_with_indicators

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_all.reshape(len(X_all), -1), y_all)
    X_resampled = X_resampled.reshape(-1, seq_length, X_all.shape[2])
    print(f"Resampled data shape: {X_resampled.shape}, {y_resampled.shape}")
    
    del X_all, y_all

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )

    del X_resampled, y_resampled, X_temp, y_temp

    n_features = X_train.shape[2]
    model, history = train_model(X_train, y_train, X_val, y_val, seq_length, n_features)
    evaluate_model(model, X_test, y_test)
    model.save('gap-d1.h5')