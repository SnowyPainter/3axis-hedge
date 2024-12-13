
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import ta
from sklearn.preprocessing import OneHotEncoder
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from collections import Counter

seqlen = 60

gap_features = ['EMA_12', 'RSI', 'ATR', 'Bollinger_band_diff', 'Volume_Change', 'Gap_Size']
def calculate_technical_indicators(df, symbol):
    df[f'{symbol}_EMA_12'] = ta.trend.EMAIndicator(df[symbol+'_Price'], window=12).ema_indicator()
    df[f'{symbol}_RSI'] = ta.momentum.RSIIndicator(df[symbol+'_Price'], window=14).rsi()
    df[f'{symbol}_ATR'] = ta.volatility.AverageTrueRange(df[symbol+'_High'], df[symbol+'_Low'], df[symbol+'_Price'], window=14).average_true_range()
    df[f'{symbol}_Bollinger_hband'] = ta.volatility.BollingerBands(df[symbol+'_Price']).bollinger_hband()
    df[f'{symbol}_Bollinger_lband'] = ta.volatility.BollingerBands(df[symbol+'_Price']).bollinger_lband()
    df[f'{symbol}_Bollinger_band_diff'] = df[f'{symbol}_Bollinger_hband'] - df[f'{symbol}_Bollinger_lband']
    df[f'{symbol}_Volume_Change'] = df[symbol+'_Volume'].pct_change().fillna(0)
    df[f'{symbol}_Gap_Size'] = df[symbol+'_Price'].pct_change().fillna(0)
    df.dropna(inplace=True)
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    scaler = StandardScaler()
    for feature in gap_features:
        df[f'{symbol}_{feature}'] = scaler.fit_transform(df[[f'{symbol}_{feature}']])
        
    return df

def predict(raw_data, symbol, model):
    df_with_indicators = calculate_technical_indicators(raw_data, symbol)
    features = ['EMA_12', 'RSI', 'ATR', 'Bollinger_band_diff', 'Volume_Change', 'Gap_Size']
    X = df_with_indicators[[f'{symbol}_{feature}' for feature in features]].tail(seqlen).values
    
    if len(X) > seqlen:
        X = X[-seqlen:]
    
    X = X.reshape(1, seqlen, len(features))
    
    prediction = model.predict(X, verbose=0)
    return prediction[0]

class GapCapture:
    
    def __init__(self, symbol, univ_model) -> None:
        self.seqlen = 60
        self.symbol = symbol
        self.univ_model = univ_model
        self.features = ['EMA_12', 'RSI', 'ATR', 'Bollinger_band_diff', 'Volume_Change', 'Gap_Size']
        raw = utils.load_historical_data(symbol, utils.today_before(700), utils.today(), interval='1d')
        
        raw = raw[raw[symbol+"_Volume"] != 0]
        self.df_with_indicators = calculate_technical_indicators(raw, symbol)

    def _finetune_model(self, model, X, y, model_name, epochs=10, batch_size=32):
        checkpoint = ModelCheckpoint(f'./trained_model/GAP_{model_name}_finetuned.h5', monitor='val_loss', save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        print(f"Fine-tuning {model_name} model...")
        history = model.fit(
            X, y, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.2, 
            callbacks=[checkpoint, early_stop]
        )
        return model, history

    def finetune_model(self):
        seq_length = self.seqlen
        
        def GAP_target_function(data, symbol, lookahead_days=1):
            """
            갭 상승 시 1, 갭 하락 시 0, 갭이 없을 시 2로 신호를 처리.
            """
            data[f'{symbol}_Gap'] = data[f'{symbol}_Price'].pct_change(lookahead_days).fillna(0)
            data[f'{symbol}_Signal'] = data[f'{symbol}_Gap'].apply(
                lambda x: 1 if x >= 0.05 else (0 if x <= -0.05 else 2)
            )
            return data
        
        symbol_data = self.df_with_indicators[[f'{self.symbol}_Price'] + [f'{self.symbol}_{feature}' for feature in self.features]].copy()
        symbol_data = GAP_target_function(symbol_data, self.symbol)
        symbol_data.dropna(inplace=True)
        
        X, y = [], []
        for i in range(len(symbol_data) - seq_length):
            X.append(symbol_data[[f'{self.symbol}_{feature}' for feature in self.features]].iloc[i:i+seq_length].values)
            y.append(symbol_data[f'{self.symbol}_Signal'].iloc[i+seq_length])
        
        if len(X) == 0 or len(y) == 0: #데이터 이상.
            return None
        
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
        encoder = OneHotEncoder(categories=[[0,1,2]], sparse=False)
        y_resampled = encoder.fit_transform(y_resampled.reshape(-1, 1))
        X_resampled = X_resampled.reshape((-1, timesteps, n_features))
        finetuned_model, history = self._finetune_model(self.univ_model, X_resampled, y_resampled, f'{self.symbol}')
        
        return finetuned_model
    
import utils
import os
from tensorflow.keras.models import load_model
symbol = ""

def load_or_finetune_gapcapture(symbol, univ_model):
    model_name = f"./trained_model/GAP_{symbol}_finetuned.h5"
    gc = GapCapture(symbol, univ_model)
    if not os.path.exists(model_name):
        return gc.finetune_model()
    return load_model_with_error_handling(model_name)

from tensorflow import keras
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

# 모델 로드 함수
def load_model_with_error_handling(model_path):
    
    with keras.utils.custom_object_scope({'focal_loss_fixed': focal_loss()}):
        loaded_model = keras.models.load_model(model_path)
        return loaded_model

symbols = ["041190.KQ", "024740.KQ", "330860.KQ", "445090.KQ"]
y = ["폭락", "폭락", "폭등", "폭등"]

# 0 폭락 1 폭등 2 변동 x 

for symbol, label in zip(symbols, y):
    print(symbol, label)
    model = load_or_finetune_gapcapture(symbol, load_model_with_error_handling("./GAP_univ.h5"))
    raw = utils.load_historical_data(symbol, utils.today_before(180), utils.today_before(2), interval='1d')
    print(raw)
    print("Prediction")
    print(predict(raw, symbol, model))