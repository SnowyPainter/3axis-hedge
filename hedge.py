import numpy as np
import pandas as pd
from scipy.optimize import minimize
import numpy as np

'''
주식 A를 헤지하기 위해서 주식 B1, B2가 사용됨.
조건: (A)와 (B1, B2)는 다른 섹터에서 상관계수가 -1에 가까워야함
조건2: 사실 뭐든 상관없음

진입: Local BNS 알고리즘에 따라 주식 A, B1, B2에 진입(매수)
포지션: 헤지 비율을 m일에 1번 계산하여 포트폴리오를 조정함
재진입: Local BNS에 따라 주식 A에 대해서 진입함
헤지 불능: B1, B2 보유량이 매우 적어지는 경우, 기존에 매도한 B1, B2에 따른 현금에 따라 Local BNS에서 B1, B2를 매수함.
포트폴리오 재조정: Local BNS에서 매수하였음에도 불구하고 B1, B2 보유량이 적은 경우, A의 비중을 줄임
--> 현금 최대 확보 기간.
'''

def calculate_beta(stock_returns, market_returns):
    cov = np.cov(stock_returns, market_returns)[0, 1]
    market_var = np.var(market_returns)
    return cov / market_var

class Axis3Hedge:
    
    def __init__(self, A, B1, B2, M,raw) -> None:
        self.symbols = [A, B1, B2, M]
        price_columns = [symbol + "_Price" for symbol in self.symbols]
        self.raw = raw[price_columns].pct_change()
        self.raw.dropna(inplace=True)
        new_column_names = ['A'] + [f'B{i}' for i in range(1, len(self.raw.columns) - 1)] + ['Market']
        self.raw.columns = new_column_names
    
    def get_hedge_ratio(self, bar, window):
        '''
        beta, vola : 음수일 때 "매수", 양수일 때 "매도"
        beta는 A가 시장리스크에 민감한 경우
        vola는 A가 변동성이 강한 경우
        '''
        raw = self.raw.iloc[bar-window:bar]
        hr = {}
        beta_A = calculate_beta(raw['A'], raw['Market'])
        beta_B1 = calculate_beta(raw['B1'], raw['Market'])
        beta_B2 = calculate_beta(raw['B2'], raw['Market'])
        hr['beta-b1'] = beta_A / beta_B1
        hr['beta-b2'] = beta_A / beta_B2

        volatility_A = raw['A'].std()
        volatility_B1 = raw['B1'].std()
        volatility_B2 = raw['B2'].std()
        rho_AB1 = np.corrcoef(raw['A'], raw['B1'])[0, 1]
        rho_AB2 = np.corrcoef(raw['A'], raw['B2'])[0, 1]
        hr['vola-b1'] = -rho_AB1 * (volatility_A / volatility_B1)
        hr['vola-b2'] = -rho_AB2 * (volatility_A / volatility_B2)
        
        return hr

    def optimize_hedging(self, A_value, B1_value, B2_value, hedge_ratios, max_risk=0.1):
        initial_weights = np.array([0, 0])
        total_portfolio_value = A_value
        risk_limit = max_risk * total_portfolio_value
        def objective(weights):
            b1_hedge_cost = weights[0] * hedge_ratios['beta-b1'] * B1_value
            b2_hedge_cost = weights[1] * hedge_ratios['beta-b2'] * B2_value
            return b1_hedge_cost + b2_hedge_cost
        constraints = [
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
            {'type': 'ineq', 'fun': lambda weights: weights[0]},  # B1 매수 비율은 0 이상
            {'type': 'ineq', 'fun': lambda weights: weights[1]},  # B2 매수 비율은 0 이상
            {'type': 'ineq', 'fun': lambda weights: risk_limit - np.abs(weights[0] * B1_value * hedge_ratios['beta-b1'])},  # B1 리스크 한도
            {'type': 'ineq', 'fun': lambda weights: risk_limit - np.abs(weights[1] * B2_value * hedge_ratios['beta-b2'])}   # B2 리스크 한도
        ]
        bounds = [(0, 1), (0, 1)] 
        optimal_solution = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return optimal_solution.x

    def portfolio_variance(self, weights, A_value, B1_value, B2_value, hedge_ratios):
        A_reduced_value = A_value * (1 - weights[0])
        B1_target_value = B1_value + (A_value - A_reduced_value) * (-hedge_ratios['vola-b1'])
        B2_target_value = B2_value + (A_value - A_reduced_value) * (-hedge_ratios['vola-b2'])
        portfolio_volatility = np.std([A_reduced_value, B1_target_value, B2_target_value])
        
        return portfolio_volatility

    def optimize_reduction(self, A_value, B1_value, B2_value, hedge_ratios):
        initial_weight = [0.1]
        optimal_solution = minimize(self.portfolio_variance, initial_weight,
                                    args=(A_value, B1_value, B2_value, hedge_ratios),
                                    bounds=[(0, 1)])
        return optimal_solution.x[0]

    def adjust_portfolio(self, A_value, B1_value, B2_value, hedge_ratios, max_risk, reduction_ratio):
        risk_limit = A_value * max_risk
        A_reduced_value = A_value * (1 - reduction_ratio)
        B1_target_value = B1_value + (A_value - A_reduced_value) * (-hedge_ratios['vola-b1'])
        B2_target_value = B2_value + (A_value - A_reduced_value) * (-hedge_ratios['vola-b2'])
        if (B1_target_value - B1_value) * hedge_ratios['vola-b1'] > risk_limit:
            B1_target_value = B1_value + risk_limit / (-hedge_ratios['vola-b1'])
        if (B2_target_value - B2_value) * hedge_ratios['vola-b2'] > risk_limit:
            B2_target_value = B2_value + risk_limit / (-hedge_ratios['vola-b2'])

        return A_reduced_value, B1_target_value, B2_target_value

'''
hr = get_hedge_ratio()

A_value = 1000
B1_value = A_value * (hr['beta-b1'] * -1 +1)
B2_value = A_value * (hr['beta-b2'] * -1 +1)
max_risk = 1
print(f"자산 A: {A_value}, 헷지용 주식 가치: {B1_value + B2_value}, 최대 허용 리스크 : {max_risk * 100} %")
hr_opt = optimize_hedging(A_value, B1_value, B2_value, hr, max_risk=max_risk)
i = 1
for h in hr_opt:
    print(f"B{i} 실제 매도/매수 비율 : {h :.2f}")
    i += 1
    
A_opt_r = optimize_reduction(A_value, B1_value, B2_value, hr)
portfolio = adjust_portfolio(A_value, B1_value, B2_value, hr, max_risk, A_opt_r)
print(portfolio)
'''