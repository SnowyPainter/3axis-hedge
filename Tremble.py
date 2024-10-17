import backtester, localbns
from tensorflow import keras
import pandas as pd
import utils

import universial_model

'''
Sideway strategy Backtest
'''

class Tremble:
    
    def __init__(self, symbol1, symbol2):
        self.buyers = {
            symbol1: keras.models.load_model(f'best_buy_{symbol1}_finetuned_model.h5'),
            symbol2: keras.models.load_model(f'best_buy_{symbol2}_finetuned_model.h5'),
        }
        self.sellers = {
            symbol1: keras.models.load_model(f'best_sell_{symbol1}_finetuned_model.h5'),
            symbol2: keras.models.load_model(f'best_sell_{symbol2}_finetuned_model.h5')
        }
        self.symbols = [symbol1, symbol2]
        
        
    def predict(self, raw_these_days, symbol):
        return localbns.predict(raw_these_days, symbol, self.buyers[symbol], self.sellers[symbol])
    
    def calculate_signal_ratio(self, buy, sell):
        epsilon = 1e-6  # 0으로 나누는 것을 방지하기 위한 작은 값
        threshold = 0.1  # 신호가 비슷한 경우를 판단하는 임계값

        if abs(buy - sell) < threshold:
            return 0, 0  # 신호가 비슷한 경우 둘 다 0으로 설정
        
        buy_ratio = buy / (buy + sell + epsilon)
        sell_ratio = sell / (buy + sell + epsilon)
        return buy_ratio, sell_ratio

    def process_signals(self, buy, sell, m=1.5):
        buy_ratio, sell_ratio = self.calculate_signal_ratio(buy, sell)
        buy_amplified = buy_ratio * m if buy_ratio > sell_ratio else buy_ratio
        sell_amplified = sell_ratio * m if sell_ratio > buy_ratio else sell_ratio

        return buy_amplified, sell_amplified
    
    def process_and_trade(self, symbol1_buy, symbol1_sell, symbol2_buy, symbol2_sell, current_ratio):
        # 신호 강도를 기반으로 매매 비율을 최적화
        symbol1_buy_ratio = symbol1_buy * current_ratio
        symbol1_sell_ratio = symbol1_sell * current_ratio
        symbol2_buy_ratio = symbol2_buy * current_ratio
        symbol2_sell_ratio = symbol2_sell * current_ratio
        
        if symbol1_buy > 0.95 and (symbol2_buy < 0.3):
            self.bt.buy(self.symbols[0], symbol1_buy_ratio)
        
        elif symbol1_sell > 0.95 and (symbol2_sell < 0.3):
            self.bt.sell(self.symbols[0], symbol1_sell_ratio)
        
        if symbol2_buy > 0.95 and (symbol1_buy < 0.3):
            self.bt.buy(self.symbols[1], symbol2_buy_ratio)
        
        elif symbol2_sell > 0.95 and (symbol1_sell < 0.3):
            self.bt.sell(self.symbols[1], symbol2_sell_ratio)
    
    def backtest(self):
        def _process_data(raw, bar):
            price_columns = list(map(lambda symbol: symbol+"_Price", self.symbols))
            return {
                "price" : raw[price_columns].iloc[bar]
            }
        raw, edited = utils.load_historical_datas(self.symbols, utils.today_before(150), utils.today_before(50), interval='1h')
        self.bt = backtester.Backtester(self.symbols, raw, 10000000000, 0.005, _process_data)
        bar = 0
        l = localbns.seqlen * 2
        while True:
            preprocessed, today = self.bt.go_next()
            if preprocessed == -1:
                break
            if bar > localbns.seqlen * 2 and bar % 2 == 0:
                these_days_data = raw.iloc[bar-(localbns.seqlen * 2):bar]
                m = 1.5
                symbol1_buy, symbol1_sell = self.predict(these_days_data, self.symbols[0])
                symbol1_buy, symbol1_sell = self.process_signals(symbol1_buy, symbol1_sell, m = m)
                symbol2_buy, symbol2_sell = self.predict(these_days_data, self.symbols[1])
                print(symbol2_buy, symbol2_sell)
                symbol2_buy, symbol2_sell = self.process_signals(symbol2_buy, symbol2_sell, m = m)
                self.process_and_trade(symbol1_buy, symbol1_sell, symbol2_buy, symbol2_sell, 0.3)
                
            bar += 1
            
            self.bt.print_stock_weights()
        
        print(self.bt.get_result())
        self.bt.plot_result(localbns.normalize(raw))

symbols = ["DAL", "XOM"]

for symbol in symbols:
    continue
    raw = utils.load_historical_data(symbol, "2022-01-01", "2024-01-01")
    d = localbns.calculate_technical_indicators(raw, symbol)
    universial_model.finetune_buy_model(symbol, d, 'LOCALBNS_buy_univ.h5')
    universial_model.finetune_sell_model(symbol, d, 'LOCALBNS_sell_univ.h5')

kw = Tremble(symbols[0], symbols[1])

kw.backtest()