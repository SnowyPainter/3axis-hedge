def min_max_normalize(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

import localbns, utils, backtester, hedge
from tensorflow import keras

class LocalBNS:
    
    def __init__(self, a_symbol, b1_symbol, b2_symbol):
        self.buyers = {}
        self.sellers = {}
        self.symbols = [a_symbol, b1_symbol, b2_symbol]
        if not utils.models_exists(self.symbols):
            for symbol in self.symbols:
                data = utils.load_historical_data(symbol, utils.today_before(1200), utils.today(), interval='1d')
                b, a = localbns.get_52_ba(data, symbol)
                self.buyers[symbol] = localbns.create_buy_model(b, symbol)
                self.sellers[symbol] = localbns.create_sell_model(a, symbol)
                self.buyers[symbol].save(f"{symbol}_Buy.keras")
                self.sellers[symbol].save(f"{symbol}_Sell.keras")
        else:
            for symbol in self.symbols:
                self.buyers[symbol] = keras.models.load_model(f'{symbol}_Buy.keras')
                self.sellers[symbol] = keras.models.load_model(f'{symbol}_Sell.keras')
            
        self.buys = []
        self.sells = []
        
    def predict(self, raw_these_days, symbol):
        return localbns.predict(raw_these_days, symbol, self.buyers[symbol], self.sellers[symbol])
    
    def get_signal_normalized(self):
        buy = min_max_normalize(self.buys[-localbns.seqlen*2:])[-1]
        sell = min_max_normalize(self.sells[-localbns.seqlen*2:])[-1]
        
        r, r2 = self.calculate_ratio(buy, sell, ignore_theshold=0.9, threshold=0.55)
        if r > 0:
            return {
                "Buy": r2,
                "Sell": 0
            }
        elif r < 0:
            return {
                "Buy": 0,
                "Sell": r2,
            }
        else:
            return {
                "Buy": 0,
                "Sell": 0
            }
        
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
    
    def backtest(self, market="^IXIC", max_risk=0.5, refresh_portfolio_period=31):
        def _process_data(raw, bar):
            price_columns = list(map(lambda symbol: symbol+"_Price", self.symbols))
            return {
                "price" : raw[price_columns].iloc[bar]
            }
        raw, edited = utils.load_historical_datas(self.symbols + [market], utils.today_before(1000), utils.today_before(50), interval='1d')
        hedge_manager = hedge.Axis3Hedge(self.symbols[0], self.symbols[1], self.symbols[2], market, raw)
        self.bt = backtester.Backtester(self.symbols, raw, 10000000000, 0.005, _process_data)
        bar = 0
        buys = {}
        sells = {}
        for symbol in self.symbols:
            buys[symbol] = []
            sells[symbol] = []
        
        hedging = False
        allow_sell = False
        allow_buy = True
        flag = False
        
        l = localbns.seqlen * 2
        while True:
            preprocessed, today = self.bt.go_next()
            if preprocessed == -1:
                break
            if bar > localbns.seqlen * 2:
                these_days_data = raw.iloc[bar-(localbns.seqlen * 2):bar]
                #getting signals - local bns
                for symbol in self.symbols:
                    buy, sell = self.predict(these_days_data, symbol)
                    buys[symbol].append(buy)
                    sells[symbol].append(sell)
                
                if len(buys[self.symbols[0]]) > l and not hedging:
                    for symbol in self.symbols:
                        buy = min_max_normalize(buys[symbol][-l:])[-1]
                        sell = min_max_normalize(sells[symbol][-l:])[-1]
                        r, r2 = self.calculate_ratio(buy, sell, ignore_theshold=0.9, threshold=0.55)
                        if r > 0 and allow_buy:
                            self.bt.buy(symbol, 0.05)
                        if r < 0 and allow_sell:
                            self.bt.sell(symbol, 0.05)
                
                A_value = self.bt.value_of(self.symbols[0])
                B1_value = self.bt.value_of(self.symbols[1])
                B2_value = self.bt.value_of(self.symbols[2])
                total_value = A_value + B1_value + B2_value
                if A_value > 0 and B1_value > 0 and B2_value > 0 and not flag:
                    hedging = True
                    allow_buy = False
                    allow_sell = False
                    flag = True
                
                if hedging and bar > 0 and bar % refresh_portfolio_period == 0:
                    print("-"*60)
                    self.bt.print_stock_weights()
                    
                    hr = hedge_manager.get_hedge_ratio(bar, localbns.seqlen * 2)
                    
                    #헷지용 주식 B1, B2에 대한 조정
                    hr_opt = hedge_manager.optimize_hedging(A_value, B1_value, B2_value, hr, max_risk)
                    print(f"자산 A: {A_value}, 헷지용 주식 가치: {B1_value + B2_value}")
                    b1_hr_opt_ratio = (B1_value * hr_opt[0]) / total_value
                    b2_hr_opt_ratio = (B2_value * hr_opt[1]) / total_value
                    if hr['vola-b1'] > 0 and abs(hr_opt[0]) > 0: #매도
                        print(f"{self.symbols[1]} 헷지 조정(Sell) {hr_opt[0]} - {b1_hr_opt_ratio}")
                        self.bt.sell(self.symbols[1], b1_hr_opt_ratio)
                    elif hr['vola-b1'] < 0 and abs(hr_opt[0]) > 0:
                        print(f"{self.symbols[1]} 헷지 조정(Buy) {hr_opt[0]} - {b1_hr_opt_ratio}")
                        self.bt.buy(self.symbols[1], b1_hr_opt_ratio)
                    
                    if hr['vola-b2'] > 0 and abs(hr_opt[1]) > 0: #매도
                        print(f"{self.symbols[2]} 헷지 조정(Sell) {hr_opt[1]} - {b2_hr_opt_ratio}")
                        self.bt.sell(self.symbols[2], b2_hr_opt_ratio)
                    elif hr['vola-b2'] < 0 and abs(hr_opt[0]) > 0:
                        print(f"{self.symbols[2]} 헷지 조정(Buy) {hr_opt[1]} - {b2_hr_opt_ratio}")
                        self.bt.buy(self.symbols[2], b2_hr_opt_ratio)
                        
                    #포트폴리오 재조정
                    A_opt_r = hedge_manager.optimize_reduction(A_value, B1_value, B2_value, hr)
                    A,B1,B2 = hedge_manager.adjust_portfolio(A_value, B1_value, B2_value, hr, max_risk, A_opt_r)
                    T = A+B1+B2
                    A_iw, B1_iw, B2_iw = A/2, B1/T, B2/T
                    A_cw, B1_cw, B2_cw = self.bt.current_weights()
                    Anw = A_iw - A_cw
                    B1nw = B1_iw - B1_cw
                    B2nw = B2_iw - B2_cw
                    if Anw < 0:
                        self.bt.sell(self.symbols[0], Anw)
                    elif Anw > 0:
                        self.bt.buy(self.symbols[0], Anw)
                        
                    if B1nw < 0:
                        self.bt.sell(self.symbols[1], B1nw)
                    elif B1nw > 0:
                        self.bt.buy(self.symbols[1], B1nw)
                        
                    if B2nw < 0:
                        self.bt.sell(self.symbols[2], B2nw)
                    elif B2nw > 0:
                        self.bt.buy(self.symbols[2], B2nw)
                    
                    
                    if A_opt_r > 0:
                        print(f"A에 대한 조정 {A_opt_r} - r {(A_value * A_opt_r) / total_value}")
                        self.bt.sell(self.symbols[0], (A_value * A_opt_r) / total_value)
                    
                    if A_value / total_value < 0.03 or B1_value / total_value < 0.03 or B2_value / total_value < 0.03:
                        print(f"더 이상 헤지할 필요 없음. 나머지 차익 실현.")
                        allow_sell = True
                        allow_buy = False
                        hedging = False
                    
            bar += 1
        
        print(self.bt.get_result())
        self.bt.plot_result(localbns.normalize(raw))
        