import backtester, localbns.localbns as localbns
from tensorflow import keras
import pandas as pd
import utils

import localbns.universial_model as universial_model

raw = utils.load_historical_data("XOM", "2022-01-01", "2024-01-01")
#raw = localbns.nplog(raw)
d = localbns.calculate_technical_indicators(raw, "XOM")
universial_model.finetune_buy_model("XOM", d, 'LOCALBNS_buy_univ.h5')
universial_model.finetune_sell_model("XOM", d, 'LOCALBNS_sell_univ.h5')

def process_data(raw, bar):
    return {
        "price": raw["XOM_Price"].iloc[bar]
    }

# Load the models
buy_model = keras.models.load_model('best_buy_XOM_finetuned_model.h5')
sell_model = keras.models.load_model('best_sell_XOM_finetuned_model.h5')

# Load historical data
symbols = ["XOM"]
start_date = utils.today_before(1000)
end_date = utils.today_before(50)
raw, _ = utils.load_historical_datas(symbols, start_date, end_date, interval='1d')
# Initialize backtester
bt = backtester.Backtester(symbols, raw, initial_amount=1000000, fee=0.005, data_proc_func=process_data)

# Backtest loop
while True:
    preprocessed, today = bt.go_next()
    if preprocessed == -1:
        break
    
    # Make prediction
    if bt.bar >= 60:
        input_data = raw.iloc[bt.bar-60:bt.bar]  # Assuming 30 days of historical data for prediction
        buy_prediction, sell_prediction = localbns.predict(input_data, "XOM", buy_model, sell_model)
        if buy_prediction > 0.5 and buy_prediction > sell_prediction:
            bt.buy("XOM", ratio=0.1)
        elif sell_prediction > 0.5 and sell_prediction > buy_prediction:
            bt.sell("XOM", ratio=0.1)

# Get and print results
results = bt.get_result()
print("Backtest Results:")
print(f"Sharpe Ratio: {results['sharp']}")
print(f"End Return: {results['end']}")
print(f"Best Return: {results['best']}")
print(f"Worst Return: {results['worst']}")
print(f"Number of Trades: {results['trades']}")
print(f"Profits: {results['profits']}")

# Plot results
bt.plot_result(localbns.normalize(raw))
