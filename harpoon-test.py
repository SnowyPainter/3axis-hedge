import utils
import harpoon_model
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

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
            start = bar - harpoon_model.seqlen - 49 # must be 49.
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
    plt.scatter(v_signals, prices[v_signals], color="green", label="V Pattern", marker="^", alpha=1, s=70)  # Increased size
    plt.scatter(inverted_v_signals, prices[inverted_v_signals], color="red", label="Inverted V Pattern", marker="v", alpha=1, s=70)  # Increased size
    plt.title(f"Price Chart with V and Inverted V Patterns: {symbol}")
    plt.xlabel("Time (Bar)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{symbol}_v_inverted_v_patterns.png")  # Save the figure as an image
    plt.close()  # Close the plot to free up memory


for symbol in ["TSLA", "NVDA", "SMCI"]:
    symbols = [symbol]
    data = utils.load_historical_for_learning(symbol, utils.today_before(900), utils.today(), interval='1d')
    if os.path.exists(f"best_HARPOON_{symbol}_finetuned_model.h5"):
        print(f"Model for {symbol} already exists. Skipping training.")
        model = load_model(f"best_HARPOON_{symbol}_finetuned_model.h5")
    else:
        model = harpoon_model.finetune_model(symbol, harpoon_model.calculate_technical_indicators(data, symbol))
    
    detect_and_plot_signals()