import utils
import harpoon_model
import backtester
import matplotlib.pyplot as plt
import numpy as np

symbol = "SMCI"
symbols = [symbol]

def _process_data(raw, bar):
    price_columns = list(map(lambda symbol: symbol+"_Price", symbols))
    return {
        "price" : raw[price_columns].iloc[bar]
    }

data = utils.load_historical_for_learning(symbol, utils.today_before(900), utils.today(), interval='1d')
model = harpoon_model.finetune_model(symbol, harpoon_model.calculate_technical_indicators(data, symbol))

def backtest():
    bt = backtester.Backtester(symbols, data, 10000000000, 0.0025, _process_data)

    bar = 0
    while True:
        preprocessed, today = bt.go_next()
        if preprocessed == -1:
            break

        if bar > harpoon_model.seqlen * 2 and bar < len(data) - harpoon_model.seqlen:
            start = bar - harpoon_model.seqlen
            end = bar
            pred = harpoon_model.predict(model, data.iloc[start:end], symbol)
            print(pred)
            if np.argmax(pred) == 1:  # 역 V자 패턴
                bt.sell(symbol, 0.1)
            elif np.argmax(pred) == 2:  # V자 패턴 
                bt.buy(symbol, 0.1)

        bar += 1

    print(bt.get_result())
    bt.plot_result(harpoon_model.normalize(data))

def detect_and_plot_signals():
    """
    V자 및 역V자 신호를 감지하고 차트 위에 시각화.
    
    :param data: 주가 데이터 (DataFrame)
    :param model: 미세 조정된 모델
    :param symbol: 분석할 주식 심볼 (문자열)
    :param harpoon_model: Harpoon 모델 객체
    """
    v_signals = []
    inverted_v_signals = []
    prices = data[f"{symbol}_Price"].values

    bar = 0
    while bar < len(data) - harpoon_model.seqlen:
        if bar > harpoon_model.seqlen * 2:
            start = bar - harpoon_model.seqlen
            end = bar
            pred = harpoon_model.predict(model, data.iloc[start:end], symbol)

            if np.argmax(pred) == 1:  # 역V자 패턴
                inverted_v_signals.append(bar)
            elif np.argmax(pred) == 2:  # V자 패턴
                v_signals.append(bar)
        
        bar += 1

    # 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label=f"{symbol} Price", color="blue")
    plt.scatter(v_signals, prices[v_signals], color="green", label="V Pattern", marker="^", alpha=0.7)
    plt.scatter(inverted_v_signals, prices[inverted_v_signals], color="red", label="Inverted V Pattern", marker="v", alpha=0.7)
    plt.title(f"Price Chart with V and Inverted V Patterns: {symbol}")
    plt.xlabel("Time (Bar)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()
    
detect_and_plot_signals()