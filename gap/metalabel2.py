import sys, os
sys.path.append('../')

import pandas as pd
import numpy as np
import os
import glob
import pickle
import ta
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from tensorflow.keras.models import load_model
import shutil


model = load_model('./GAP_univ.h5')
seqlen = 90
def predict(df_with_indicators, symbol):
    features = ['EMA_12', 'RSI', 'ATR', 'Bollinger_band_diff', 'Volume_Change']
    predictions = []

    for i in range(len(df_with_indicators) - seqlen + 1):
        X = df_with_indicators[[f'{symbol}_{feature}' for feature in features]].iloc[i:i + seqlen].values
        X = X.reshape(1, seqlen, len(features))
        prediction = model.predict(X, verbose=0)[0] 
        predictions.append(np.argmax(prediction))
    
    return np.array(predictions)

class MetaLabelingRandomForest:
    def __init__(self, seq_length=90):
        self.seq_length = seq_length
        self.gap_features = [
            'EMA_12', 'RSI', 'ATR', 'Bollinger_band_diff', 'Volume_Change', 
            'Gap_Size', 'MACD', 'Stoch', 'WilliamsR', 'Price_ROC', 
            'CCI', 'ADX', 'Momentum', 'DMI', 'VWAP', 'Meta'
        ]

        #Meta 만 따로 normalized
        self.gap_features_normalized = [
            'EMA_12', 'RSI', 'ATR', 'Bollinger_band_diff', 'Volume_Change',
            'MACD', 'Stoch', 'WilliamsR', 'Price_ROC', 'CCI', 'ADX', 'DMI', 'VWAP', 'Momentum'
        ]

        self.gap_features_learning = [
            'Meta', 'ATR', 'VWAP', 'EMA_12', 'Bollinger_band_diff', 'Price_ROC'
        ]

    def calculate_technical_indicators(self, df, symbol):
        """Calculate technical indicators for a given symbol."""
        df[f'{symbol}_EMA_12'] = ta.trend.EMAIndicator(df[symbol+'_Open'], window=12).ema_indicator()
        df[f'{symbol}_RSI'] = ta.momentum.RSIIndicator(df[symbol+'_Open'], window=14).rsi()
        df[f'{symbol}_ATR'] = ta.volatility.AverageTrueRange(df[symbol+'_High'], df[symbol+'_Low'], df[symbol+'_Open'], window=14).average_true_range()
        df[f'{symbol}_Bollinger_hband'] = ta.volatility.BollingerBands(df[symbol+'_Open']).bollinger_hband()
        df[f'{symbol}_Bollinger_lband'] = ta.volatility.BollingerBands(df[symbol+'_Open']).bollinger_lband()
        df[f'{symbol}_Bollinger_band_diff'] = df[f'{symbol}_Bollinger_hband'] - df[f'{symbol}_Bollinger_lband']
        df[f'{symbol}_Volume_Change'] = df[symbol+'_Volume'].pct_change().fillna(0)
        df[f'{symbol}_Gap_Size'] = (df[symbol+'_High'] - df[symbol+'_Open']) / df[symbol+'_Open']
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

        df.dropna(inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)

        # Normalize features
        scaler = StandardScaler()
        for feature in self.gap_features_normalized:
            df[f'{symbol}_{feature}'] = scaler.fit_transform(df[[f'{symbol}_{feature}']])

        predictions = predict(df, symbol)
        df[f'{symbol}_Meta'] = np.nan  # Initialize with NaN
        df[f'{symbol}_Meta'].iloc[self.seq_length-1:len(predictions)+self.seq_length-1] = predictions  # Assign predictions

        df.dropna(inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)

        df[f'{symbol}_Meta'] = scaler.fit_transform(df[[f'{symbol}_Meta']])

        return df
    
    def gap_target_function(self, data, symbol, lookahead_days=1):
        """Create signal for meta labeling"""
        data[f'{symbol}_Signal'] = data.apply(
            lambda row: 1 if (row[f'{symbol}_Gap_Size'] > 0.005)
                             else 0, axis=1
        )
        return data

        
    def save_data_chunk(self, X, y, prefix, chunk_dir='./chunks'):
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_id = len(glob.glob(os.path.join(chunk_dir, f'{prefix}data_chunk_*.pkl')))
        with open(os.path.join(chunk_dir, f'{prefix}data_chunk_{chunk_id}.pkl'), 'wb') as f:
            pickle.dump((np.array(X), np.array(y)), f)

    def process_symbol_data(self, symbol, df_with_indicators):
        """Process data for a single symbol"""
        print(f"Processing {symbol}...")
        data = df_with_indicators[[f'{symbol}_Open'] + [f'{symbol}_{feature}' for feature in self.gap_features]].copy()
        
        data = self.gap_target_function(data, symbol)
        data.dropna(inplace=True)
        
        symbol_X, symbol_y = [], []
        for i in range(len(data) - self.seq_length):
            symbol_X.append(data[[f'{symbol}_{feature}' for feature in self.gap_features_learning]].iloc[i:i+self.seq_length].mean().values)
            symbol_y.append(data[f'{symbol}_Signal'].iloc[i+self.seq_length])
        
        print(f"{symbol} : Preprocessed")
        
        ones_count = sum(1 for i in symbol_y if i == 1)
        zeros_count = sum(1 for i in symbol_y if i == 0)
        print(f"{symbol} : Number of 1's: {ones_count}, Number of 0's: {zeros_count}")
        
        return symbol_X, symbol_y

    def load_dataset(self):
        X_all, y_all = [], []
        for chunk_file in glob.glob(os.path.join('./chunks', f'meta_gap_data_chunk_*.pkl')):
            with open(chunk_file, 'rb') as f:
                X_chunk, y_chunk = pickle.load(f)
                X_all.append(X_chunk)
                y_all.append(y_chunk)
        
        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=1e10, neginf=-1e10)

        return X_all, y_all

    def create_dataset(self, symbols, df_with_indicators):
        if os.path.exists('./chunks'):
            shutil.rmtree('./chunks')

        """Create full dataset from multiple symbols"""
        X, y = [], []
        for symbol in symbols:
            symbol_X, symbol_y = self.process_symbol_data(symbol, df_with_indicators)
            X.extend(symbol_X)
            y.extend(symbol_y)

            if len(X) > 4000:  # Adjust this threshold as needed
                self.save_data_chunk(X, y, "meta_gap_")
                X, y = [], []
        
        return np.array(X), np.array(y)
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest Classifier"""
        rf_classifier = RandomForestClassifier(
            n_estimators=100, 
            max_depth=30, 
            min_samples_split=10, 
            min_samples_leaf=4, 
            random_state=42, 
            class_weight='balanced'
        )
        
        rf_classifier.fit(X_train, y_train)
        
        joblib.dump(rf_classifier, 'meta_gap_random_forest.joblib')
        
        return rf_classifier

    def evaluate_model(self, model, X_test, y_test):
        import features

        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        original_accuracy = accuracy_score(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        features.features_MDA_importances(X_test, y_test, model, original_accuracy, self.gap_features_learning)

    def run_meta_labeling(self, combined_prices, symbols):
        """Main method to run meta labeling"""
        df_with_indicators = combined_prices.copy()
        if os.path.exists('sp500_combined_prices_with_indicators_GAP.pkl'):
            df_with_indicators = pd.read_pickle('sp500_combined_prices_with_indicators_GAP.pkl')
            print("Loaded existing DataFrame from 'sp500_combined_prices_with_indicators_GAP.pkl'")
        else: # Add indicators if not already present
            new_indicators = {}
            for stock in symbols:
                temp_df = pd.DataFrame({
                    f'{stock}_Open': df_with_indicators[f"{stock}_Open"],
                    f'{stock}_Close': df_with_indicators[f"{stock}_Close"],
                    f'{stock}_High': df_with_indicators[f"{stock}_High"],
                    f'{stock}_Low': df_with_indicators[f"{stock}_Low"],
                    f'{stock}_Volume' : df_with_indicators[f"{stock}_Volume"]
                })
                temp_df = self.calculate_technical_indicators(temp_df, stock)
                for indicator in self.gap_features:
                    new_indicators[f'{stock}_{indicator}'] = temp_df[f'{stock}_{indicator}']
            
            df_with_indicators = pd.concat([df_with_indicators, pd.DataFrame(new_indicators)], axis=1)
            df_with_indicators.to_pickle('sp500_combined_prices_with_indicators_GAP.pkl')
        
        #if not glob.glob('./chunks/meta_gap_data_chunk_*.pkl'):
        X, y = self.create_dataset(symbols, df_with_indicators)

        X, y = self.load_dataset()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_model = self.train_random_forest(X_train, y_train)
        self.evaluate_model(rf_model, X_test, y_test)

# Example usage
if __name__ == "__main__":
    combined_prices = pd.read_pickle('sp500_combined_close_volume_prices.pkl')
    symbols = list(set(col.split('_')[0] for col in combined_prices.columns))
    
    meta_labeler = MetaLabelingRandomForest()
    meta_labeler.run_meta_labeling(combined_prices, symbols)