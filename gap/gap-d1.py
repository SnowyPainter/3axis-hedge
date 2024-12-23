import sys, os
sys.path.append('../')
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, LayerNormalization, Add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

features = [
    'EMA_12', 'RSI', 'ATR', 'Bollinger_band_diff', 'Volume_Change', 'Gap_Size'
]

features_normalized = [
    'EMA_12', 'RSI', 'ATR', 'Bollinger_band_diff', 'Volume_Change'
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
    lstm_out1 = LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
    attention_scores = Dense(1, activation='tanh')(lstm_out1)
    attention_weights = tf.nn.softmax(attention_scores, axis=1)
    attention_output = tf.multiply(lstm_out1, attention_weights)
    attention_output = tf.reduce_sum(attention_output, axis=1)
    dense1 = Dense(128, activation='relu')(attention_output)
    dropout1 = Dropout(0.3)(dense1)
    dense2 = Dense(64, activation='relu')(dropout1)
    output = Dense(n_classes, activation='softmax')(dense2)
    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer='adam',
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
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Model training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
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

if __name__ == "__main__":
    seq_length = 45
    
    combined_prices = pd.read_pickle('sp500_combined_close_volume_prices.pkl')
    symbols = list(set(col.split('_')[0] for col in combined_prices.columns))
    df_with_indicators = pd.read_pickle(f'./gap-light-df-indicator-{seq_length}.pkl')
    
    X_all, y_all = [], []
    for symbol in symbols:
        X_symbol, y_symbol = create_features_and_labels(df_with_indicators, symbol, seq_length)
        X_all.append(X_symbol)
        y_all.append(y_symbol)
    
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_all.reshape(len(X_all), -1), y_all)
    X_resampled = X_resampled.reshape(-1, seq_length, X_all.shape[2])
    print(f"Resampled data shape: {X_resampled.shape}, {y_resampled.shape}")
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )

    n_features = X_train.shape[2]
    model, history = train_model(X_train, y_train, X_val, y_val, seq_length, n_features)
    evaluate_model(model, X_test, y_test)
    model.save('gap-d1.h5')