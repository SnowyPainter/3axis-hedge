def min_max_normalize(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

import os
import localbns.localbns as localbns, utils, backtester, localbns.hedge as hedge
from tensorflow import keras

class tester:
    
    def __init__(self, symbol):
        import localbns.universial_model as universial_model
        raw = utils.load_historical_data(symbol, "2022-01-01", "2024-01-01")
        d = localbns.calculate_technical_indicators(raw, symbol)
        
        if not os.path.exists(f'best_buy_{symbol}_finetuned_model.keras'):
            universial_model.finetune_buy_model(symbol, d, 'LOCALBNS_buy_univ.keras')
        if not os.path.exists(f'best_sell_{symbol}_finetuned_model.keras'):
            universial_model.finetune_sell_model(symbol, d, 'LOCALBNS_sell_univ.keras')

        self.buyers = {
            symbol: keras.models.load_model(f'best_buy_{symbol}_finetuned_model.keras')
        }
        self.sellers = {
            symbol: keras.models.load_model(f'best_sell_{symbol}_finetuned_model.keras')
        }
        self.symbols = [symbol]
        
        self.buys = []
        self.sells = []
        
    def predict(self, raw_these_days, symbol):
        return localbns.predict(raw_these_days, symbol, self.buyers[symbol], self.sellers[symbol])

    def similarity_ratio(self, buyer, seller):
        difference = abs(buyer - seller)
        max_difference = 1.0
        ratio = 1 - (difference / max_difference)
        return ratio
    def calculate_ratio(self, buyer, seller, ignore_theshold = 0.9, threshold=0.8):
        if self.similarity_ratio(buyer, seller) > ignore_theshold: #둘다 높은 경우
            return 0, 0 #don't do
        elif seller > buyer and seller / (buyer + seller) > threshold:
            return -1, seller / (buyer + seller)
        elif buyer > seller and  buyer / (buyer + seller) > threshold:
            return 1, buyer / (buyer + seller)
        else:
            return 0, 0
    
    def backtest(self):
        def _process_data(raw, bar):
            price_columns = list(map(lambda symbol: symbol+"_Price", self.symbols))
            return {
                "price" : raw[price_columns].iloc[bar]
            }
        raw, edited = utils.load_historical_datas(self.symbols, utils.today_before(365), utils.today_before(0), interval='1d')
        self.bt = backtester.Backtester(self.symbols, raw, 10000000000, 0.005, _process_data)
        bar = 0
        buys = {}
        sells = {}
        for symbol in self.symbols:
            buys[symbol] = []
            sells[symbol] = []
        
        while True:
            preprocessed, today = self.bt.go_next()
            if preprocessed == -1:
                break
            if bar > localbns.seqlen:
                raw_these_days = raw.iloc[bar-localbns.seqlen:bar]
                buy, sell = self.predict(raw_these_days, self.symbols[0])
                sim = self.similarity_ratio(buy, sell)
                signal, p = self.calculate_ratio(buy, sell)
                
                if sim <= 0.6:
                    if signal == -1:
                        self.bt.sell(self.symbols[0], 0.01)
                    elif signal == 1:
                        self.bt.buy(self.symbols[0], 0.01)

            bar += 1
        
        print(self.bt.get_result())
        self.bt.plot_result(localbns.normalize(raw))

t = tester("TSLA")
t.backtest()