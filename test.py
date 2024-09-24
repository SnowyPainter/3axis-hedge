import localbns_backtester
import hedge
A = "DAL" #델타항공
B1 = "XOM"
B2 = "SHEL"

lbns = localbns_backtester.LocalBNS(A, B1, B2)
lbns.backtest(max_risk=0.3, refresh_portfolio_period=31, A_ratio=20)