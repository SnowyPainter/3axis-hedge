import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from collections import namedtuple
import pytz
import os, re

def today(tz = 'Asia/Seoul'):
        return datetime.now(pytz.timezone(tz))
def today_before(day, tz = 'Asia/Seoul'):
    return datetime.now(pytz.timezone(tz)) - timedelta(days=day)

def load_historical_data(symbol, start, end, interval='1d'):
    d = yf.download(symbol, start=start, end=end, interval=interval)
    d.rename(columns={'Close': symbol+'_Price', 'Volume' : symbol+"_Volume", 'High' : symbol+"_High", 'Low' : symbol+"_Low"}, inplace=True)
    d.index = pd.to_datetime(d.index, format="%Y-%m-%d %H:%M:%S%z")
    return d[[symbol+'_Price', symbol+"_Volume", symbol+"_High", symbol+"_Low"]]

def merge_dfs(dfs):
    if len(dfs) == 0:
        return pd.DataFrame({})
    merged = dfs[0]
    for i in range(1, len(dfs)):
        merged = merged.join(dfs[i])
    return merged

def load_historical_datas(symbols, start, end, interval='1d'):
    dfs = []
    edit_symbols = symbols.copy()
    for symbol in symbols:
        df = load_historical_data(symbol, start, end, interval)
        if df.empty == True:
            edit_symbols.remove(symbol)
            continue
        dfs.append(df)
    
    merged = merge_dfs(dfs)
    merged.dropna(inplace=True)
    
    return merged, edit_symbols

def models_exists(symbols):
    flag = True
    for symbol in symbols:
        flag = os.path.exists(f"{symbol}_Buy.keras")
        flag = os.path.exists(f"{symbol}_Sell.keras")
    return flag