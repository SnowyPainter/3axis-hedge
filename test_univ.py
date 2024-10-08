import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os

import localbns
import utils
from universial_model import finetune_buy_model, finetune_sell_model
from localbns_backtester import LocalBNS

def load_and_prepare_data(symbol, start_date, end_date):
    raw = utils.load_historical_data(symbol, start_date, end_date, interval='1d')
    df = localbns.normalize(raw).copy()
    df = localbns.calculate_technical_indicators(df, symbol)
    return df

def main():
    # Perform hedging backtest using finetuned models for XOM, SHEL, and DAL
    symbols = ['XOM', 'SHEL', 'DAL']
    
    # Load finetuned models for each symbol
    finetuned_models = {}
    for symbol in symbols:
        
        if not os.path.exists(f'buy_model_{symbol}_finetuned.h5') or not os.path.exists(f'sell_model_{symbol}_finetuned.h5'):
            start_date = utils.today_before(365 * 2)  # 2년치 데이터
            end_date = utils.today()
            df = load_and_prepare_data(symbol, start_date, end_date)    
        if not os.path.exists(f'buy_model_{symbol}_finetuned.h5'):
            finetuned_buy_model = finetune_buy_model(symbol, df, original_model_path='buy_model_univ.h5')
            finetuned_buy_model.save(f'buy_model_{symbol}_finetuned.h5')
            print(f"Finetuned buy model saved as 'buy_model_{symbol}_finetuned.h5'")
        if not os.path.exists(f'sell_model_{symbol}_finetuned.h5'):
            finetuned_sell_model = finetune_sell_model(symbol, df, original_model_path='sell_model_univ.h5')
            finetuned_sell_model.save(f'sell_model_{symbol}_finetuned.h5')
            print(f"Finetuned sell model saved as 'sell_model_{symbol}_finetuned.h5'")
        
        finetuned_models[symbol] = {
            'buy': load_model(f'buy_model_{symbol}_finetuned.h5'),
            'sell': load_model(f'sell_model_{symbol}_finetuned.h5')
        }
    
    bns = LocalBNS(symbols[0], symbols[1], symbols[2]) 
    bns.backtest(max_risk=1, A_ratio=25)

if __name__ == "__main__":
    main()
