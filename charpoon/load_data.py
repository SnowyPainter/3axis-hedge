import sys, os
sys.path.append('../')

import utils
import pandas as pd
import glob
import matplotlib.pyplot as plt

def create_pickle():
    crypto_list = pd.read_csv('../binance-crypto-usd.csv')

    crypto_symbols = crypto_list['Symbol'] + '-' + crypto_list['QuoteAsset']
    symbols = crypto_symbols.head(15)
    combined_df = pd.DataFrame()

    for symbol in symbols:
        df = utils.load_historical_for_learning(symbol, utils.today_before(700), utils.today(), interval='1h')
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

create_pickle()