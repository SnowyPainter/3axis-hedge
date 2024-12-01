import sys, os
sys.path.append('../')

import utils
import pandas as pd

def create_pickle():
    crypto_list = pd.read_csv('./binance-cryptos.csv')

    crypto_symbols = crypto_list['BaseAsset'] + '-' + crypto_list['QuoteAsset']
    symbols = crypto_symbols.head(15)
    combined_df = pd.DataFrame()

    for symbol in symbols:
        df = utils.load_historical_for_learning(symbol, utils.today_before(8), utils.today(), interval='1m')
        if df.empty:
            continue

        if combined_df.empty:
            combined_df = df
        else:
            combined_df = combined_df.join(df, how='outer')

    combined_df.sort_index(inplace=True)
    original_columns = len(combined_df.columns)
    combined_df = combined_df.dropna(axis=1, how='all')
    removed_columns = original_columns - len(combined_df.columns)
    print(f"Removed {removed_columns} columns where all values were NaN.")
    print(f"Remaining columns: {len(combined_df.columns)}")
    combined_df.dropna(inplace=True)
    combined_df.to_pickle('./crypto-ohlcv.pkl')

def load_pickle():
    df = pd.read_pickle('./crypto-ohlcv.pkl')
    return df

load_pickle().to_csv('./crypto-ohlcv.csv')