import utils
import harpoon_model
import backtester

import numpy as np

symbol = "SMCI"
symbols = [symbol]

def _process_data(raw, bar):
    price_columns = list(map(lambda symbol: symbol+"_Price", symbols))
    return {
        "price" : raw[price_columns].iloc[bar]
    }

data = utils.load_historical_for_learning(symbol, utils.today_before(900), utils.today(), interval='1d')
model = harpoon_model.finetune_model(symbol, harpoon_model.calculate_technical_indicators(data))

bt = backtester.Backtester(symbols, data, 10000000000, 0.005, _process_data)

bar = 0
while True:
    preprocessed, today = bt.go_next()
    if preprocessed == -1:
        break

    if bar > harpoon_model.seqlen * 2 and bar < len(data) - harpoon_model.seqlen:
        start = bar
        end = bar + harpoon_model.seqlen
        pred = harpoon_model.predict(model, data.iloc[start:end], symbol)
        print(pred)
        if np.argmax(pred) == 1:  # 역 V자 패턴
            bt.sell(symbol, 0.1)
        elif np.argmax(pred) == 2:  # V자 패턴 
            bt.buy(symbol, 0.1)

    bar += 1

print(bt.get_result())
bt.plot_result(harpoon_model.normalize(data))