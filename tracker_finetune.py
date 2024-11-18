import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import ta
from tensorflow import keras
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from collections import Counter
from tensorflow.keras.utils import to_categorical

import utils

seqlen = 12

track_features = ['EMA_12', 'MACD_diff', 'ATR', 'Bollinger_band_diff', 'RSI', 'Momentum', 'Volume_Change']
def calculate_technical_indicators(df, symbol):
    """
    주어진 데이터프레임에 기술 지표를 계산하여 추가합니다.
    """
    # 지수 이동 평균 (EMA) - window=12시간 대신 3~5시간 설정으로 조정
    df[f'{symbol}_EMA_12'] = ta.trend.EMAIndicator(df[symbol+'_Price'], window=5).ema_indicator()

    # MACD와 Signal 차이 (MACD_diff) - 빠른 신호와 느린 신호 모두 5시간 이내로 설정
    macd = ta.trend.MACD(df[symbol+'_Price'], window_slow=5, window_fast=3, window_sign=3)
    df[f'{symbol}_MACD_diff'] = macd.macd_diff()

    # 평균 진폭 (ATR) - 5시간 이내로 변동성 조정
    df[f'{symbol}_ATR'] = ta.volatility.AverageTrueRange(
        high=df[symbol+'_High'], low=df[symbol+'_Low'], close=df[symbol+'_Price'], window=5
    ).average_true_range()

    # 볼린저 밴드 차이 (Bollinger_band_diff) - 짧은 기간 변동성 캡처
    bollinger = ta.volatility.BollingerBands(df[symbol+'_Price'], window=5)
    df[f'{symbol}_Bollinger_band_diff'] = bollinger.bollinger_hband() - bollinger.bollinger_lband()

    # 상대 강도 지수 (RSI) - 매우 짧은 변동성 대응을 위해 5시간 설정
    df[f'{symbol}_RSI'] = ta.momentum.RSIIndicator(df[symbol+'_Price'], window=5).rsi()

    # 모멘텀 지표 (Momentum) - Williams %R 대신 ROC로 대체
    df[f'{symbol}_Momentum'] = ta.momentum.ROCIndicator(df[symbol+'_Price'], window=5).roc()

    # 거래량 변화율 (Volume_Change) - 거래량 증감율
    df[f'{symbol}_Volume_Change'] = df[symbol+'_Volume'].pct_change().fillna(0)

    # 결측값 제거
    df.dropna(inplace=True)

    # 표준화
    for feature in track_features:
        df[f'{symbol}_{feature}'] = (df[f'{symbol}_{feature}'] - df[f'{symbol}_{feature}'].mean()) / df[f'{symbol}_{feature}'].std()

    return df

def predict(raw_data, symbol, model):
    df_with_indicators = calculate_technical_indicators(raw_data, symbol)
    X = df_with_indicators[[f'{symbol}_{feature}' for feature in track_features]].tail(seqlen).values
    
    if len(X) > seqlen:
        X = X[-seqlen:]
    
    X = X.reshape(1, seqlen, len(track_features))
    
    prediction = model.predict(X, verbose=0)
    return prediction[0]

def normalize(df):
    range_val = df.max() - df.min()
    range_val[range_val == 0] = 1
    return (df - df.min()) / (range_val)

class Tracker:
    
    def __init__(self, symbol, univ_model) -> None:
        self.seqlen = 12
        self.symbol = symbol
        self.univ_model = univ_model
        self.features = track_features
        raw = utils.load_historical_data(symbol, utils.today_before(700), utils.today(), interval='1h')
        
        raw = raw[raw[symbol+"_Volume"] != 0]
        
        self.df_with_indicators = calculate_technical_indicators(raw, symbol)

    def _finetune_model(self, model, X, y, model_name, epochs=10, batch_size=48):
        checkpoint = ModelCheckpoint(f'./trained_model/TRACK_{model_name}_finetuned.h5', monitor='val_loss', save_best_only=True, mode='min')
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
        
        def TRACK_target_function(data, symbol, lookahead_hours=3, threshold=0.005):
            """
            상승/하락/변동 없음으로 3개의 클래스로 분류합니다.
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
            print(data[f'{symbol}_Signal'])
            
            return data
        
        symbol_data = self.df_with_indicators[[f'{self.symbol}_Price'] + [f'{self.symbol}_{feature}' for feature in self.features]].copy()
        symbol_data = TRACK_target_function(symbol_data, self.symbol)
        symbol_data.dropna(inplace=True)
        
        X, y = [], []
        for i in range(len(symbol_data) - seq_length):
            X.append(symbol_data[[f'{self.symbol}_{feature}' for feature in self.features]].iloc[i:i+seq_length].values)
            y.append(symbol_data[f'{self.symbol}_Signal'].iloc[i+seq_length])
        
        if len(X) == 0 or len(y) == 0:  # 데이터 이상 처리
            return None
        
        # 데이터를 다중 클래스 문제에 맞게 One-Hot Encoding
        X = np.array(X)
        y = np.array(y)

        encoder = OneHotEncoder(sparse=False)
        y = encoder.fit_transform(y.reshape(-1, 1))
        
        # SMOTE를 사용한 오버샘플링
        n_samples, timesteps, n_features = X.shape
        X_flat = X.reshape((n_samples, timesteps * n_features))
        
        smote = SMOTE(k_neighbors=3, random_state=42)
        smote_tomek = SMOTETomek(smote=smote, random_state=42)
        X_resampled, y_resampled = smote_tomek.fit_resample(X_flat, y.argmax(axis=1))  # 다중 클래스 대응
        
        # 원래 형태로 복원
        X_resampled = X_resampled.reshape((-1, timesteps, n_features))
        y_resampled = to_categorical(y_resampled, num_classes=3)
        
        finetuned_model, history = self._finetune_model(self.univ_model, X_resampled, y_resampled, f'{self.symbol}')
        
        return finetuned_model

t = Tracker("389020.KQ", keras.models.load_model("TRACK_univ.h5"))
t.finetune_model()