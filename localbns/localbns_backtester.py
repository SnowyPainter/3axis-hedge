def min_max_normalize(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

import localbns.localbns as localbns, utils, backtester, localbns.hedge as hedge
from tensorflow import keras

class LocalBNS:
    
    def __init__(self, a_symbol, b1_symbol, b2_symbol):
        self.buyers = {
            a_symbol: keras.models.load_model(f'best_buy_{a_symbol}_finetuned_model.h5'),
            b1_symbol: keras.models.load_model(f'best_buy_{b1_symbol}_finetuned_model.h5'),
            b2_symbol: keras.models.load_model(f'best_buy_{b2_symbol}_finetuned_model.h5')
        }
        self.sellers = {
            a_symbol: keras.models.load_model(f'best_sell_{a_symbol}_finetuned_model.h5'),
            b1_symbol: keras.models.load_model(f'best_sell_{b1_symbol}_finetuned_model.h5'),
            b2_symbol: keras.models.load_model(f'best_sell_{b2_symbol}_finetuned_model.h5')
        }
        self.symbols = [a_symbol, b1_symbol, b2_symbol]
        
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
    
    def backtest(self, market="^IXIC", max_risk=0.5, refresh_portfolio_period=31, A_ratio=20):
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
        start_day = None
        left_r = (100 - A_ratio) / 2
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
                
                #매집
                a, b1, b2 = self.bt.current_weights()
                if len(buys[self.symbols[0]]) > l and not hedging:
                    for symbol in self.symbols:
                        buy = min_max_normalize(buys[symbol][-l:])[-1]
                        sell = min_max_normalize(sells[symbol][-l:])[-1]
                        r, r2 = self.calculate_ratio(buy, sell, ignore_theshold=0.9, threshold=0.55)
                        if not flag: #still 매집 기간
                            if a >= A_ratio and symbol == self.symbols[0]: #a 주식 매수 제한
                                continue
                            if b1 >= left_r and symbol == self.symbols[1]:
                                continue
                            if b2 >= left_r and symbol == self.symbols[2]: 
                                continue
                        if r > 0 and allow_buy:
                            self.bt.buy(symbol, r2 / 10)
                        if r < 0 and allow_sell:
                            self.bt.sell(symbol, r2 / 10)
                
                A_value = self.bt.value_of(self.symbols[0])
                B1_value = self.bt.value_of(self.symbols[1])
                B2_value = self.bt.value_of(self.symbols[2])
                total_value = A_value + B1_value + B2_value
                if A_value > 0 and B1_value > 0 and B2_value > 0 and not flag:
                    hr = hedge_manager.get_hedge_ratio(bar, localbns.seqlen * 2)
                    b1_h = abs(hr['vola-b1'])
                    b2_h = abs(hr['vola-b2'])
                    m1 = max_risk / b1_h
                    m2 = max_risk / b2_h
                    if a > A_ratio and ((b1 > (100-a)/2 or b2 > (100-a)/2) and b1 + b2 > (80-a)/2):
                    #if 1 - max_risk / 2 < m1 < 1 + max_risk / 2 and 1 - max_risk / 2 < m2 < 1 + max_risk / 2 and (a+b1+b2) > 80:
                        print("헷지 시작")
                        self.bt.print_stock_weights()
                        hedging = True
                        allow_buy = False
                        allow_sell = False
                        flag = True
                        start_day = today
                
                if hedging and bar > 0 and bar % refresh_portfolio_period == 0:
                    print("-"*60)
                    self.bt.print_stock_weights()
                    hr = hedge_manager.get_hedge_ratio(bar, localbns.seqlen * 2)
                    
                    #헷지용 주식 B1, B2에 대한 조정
                    hr_opt = hedge_manager.optimize_hedging(A_value, B1_value, B2_value, hr, max_risk)
                    for i in range(0, 2):
                        v = self.bt.value_of(self.symbols[i+1])
                        indic = hr[f'vola-b{i+1}'] # True == 매도
                        indic2 = abs(hr_opt[i])
                        hr_opt_ratio = (v * hr_opt[i]) / total_value
                        s = self.symbols[i+1]
                        if indic > 0 and indic2 > 0: #매도
                            print(f"헷지 {s} : {hr_opt_ratio * 100 :.2f} % 매도")
                            self.bt.sell(s, hr_opt_ratio)
                        elif indic < 0 and indic2 > 0: #매수
                            print(f"헷지 {s} : {hr_opt_ratio * 100 :.2f} % 매수")
                            self.bt.buy(s, hr_opt_ratio)

                    #포트폴리오 재조정
                    A_opt_r = hedge_manager.optimize_reduction(A_value, B1_value, B2_value, hr)
                    A,B1,B2 = hedge_manager.adjust_portfolio(A_value, B1_value, B2_value, hr, max_risk, A_opt_r)
                    T = A+B1+B2
                    print("포트폴리오 재조정")
                    for i, iw, cw in zip(range(3), [A/T, B1/T, B2/T], self.bt.current_weights()):
                        cw /= 100
                        nw = iw - cw
                        r = abs(nw) * cw
                        s = self.symbols[i]
                        if nw < 0:
                            print(f"{s} : {abs(nw) * 100 :.2f}({r * 100 :.2f}) % 매도 | ", end='')
                            self.bt.sell(s, r)
                        elif nw > 0:
                            print(f"{s} : {abs(nw) * 100 :.2f}({r * 100 :.2f}) % 매수 | ", end='')
                            self.bt.buy(self.symbols[i], r)
                    print('')
                    #헤지용 자산 잔고 바닥임을 감안하여 A 조정
                    if A_opt_r > 0:
                        r = (A_value * A_opt_r) / total_value
                        print(f"(A)매도 조정 - {self.symbols[0]} : {r * 100 :.2f} %")
                        self.bt.sell(self.symbols[0], (A_value * A_opt_r) / total_value)
                    
                    #헤지용 자산 잔고 바닥
                    if B1_value / total_value < 0.03 or B2_value / total_value < 0.03:
                        print(f"헤지 불능 상태")
                        self.bt.print_stock_weights()
                        allow_sell = True
                        allow_buy = False
                        hedging = False
                    
            bar += 1
        
        print(f"{start_day} 부터 {raw.index[-1]} 까지의 백테스팅")
        
        print(self.bt.get_result())
        self.bt.plot_result(localbns.normalize(raw))
        