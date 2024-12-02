import sys, os
sys.path.append('../')

import utils
import univ_model
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.cluster import DBSCAN

def detect_and_plot_signals():
    sell_signals = []
    buy_signals = []
    prices = data[f"{symbol}_Price"].values
    bar = 0
    # 신호 감지
    while bar < len(data) - univ_model.seq_length:
        if bar > univ_model.seq_length * 3:
            start = bar - univ_model.seq_length - 164
            end = bar
            pred = univ_model.predict(model, data.iloc[start:end], symbol)
            if np.argmax(pred) == 1:
                buy_signals.append(bar)
            elif np.argmax(pred) == 2:
                sell_signals.append(bar)
        
        bar += 1
    
    # 신호 군집화
    def cluster_signals(signal_indices):
        if not signal_indices:
            return []
        clustering = DBSCAN(eps=10, min_samples=2).fit(np.array(signal_indices).reshape(-1, 1))
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label == -1:
                continue  # Noise 제거
            clusters.setdefault(label, []).append(signal_indices[idx])
        avg_prices = [
            (np.mean(cluster), np.mean(prices[cluster])) for cluster in clusters.values()
        ]
        return avg_prices
    
    sell_clusters = cluster_signals(sell_signals)
    buy_clusters = cluster_signals(buy_signals)
    
    # 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label=f"{symbol} Price", color="blue")
    plt.scatter(sell_signals, prices[sell_signals], color="red", label="Sell", marker="v", alpha=1, s=70)
    plt.scatter(buy_signals, prices[buy_signals], color="green", label="Buy", marker="^", alpha=1, s=70)
    
    for avg_bar, avg_price in buy_clusters:
        plt.axhline(y=avg_price, color="green", linestyle="--", alpha=0.7, label="Avg Buy Price")
    for avg_bar, avg_price in sell_clusters:
        plt.axhline(y=avg_price, color="red", linestyle="--", alpha=0.7, label="Avg Sell Price")
    
    # 기타 설정
    plt.title(f"Price Chart with Signals: {symbol}")
    plt.xlabel("Time (Bar)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{symbol}.png")
    plt.close()

for symbol in ["BTC-USD"]:
    symbols = [symbol]
    data = utils.load_historical_for_learning(symbol, utils.today_before(8), utils.today(), interval='1m')
    if os.path.exists(f"best_602o_{symbol}_finetuned_model.h5"):
        print(f"Model for {symbol} already exists. Skipping training.")
        model = load_model(f"best_602o_{symbol}_finetuned_model.h5")
    else:
        model = univ_model.finetune_model(symbol, univ_model.calculate_technical_indicators(data, symbol))
    
    data = utils.load_historical_for_learning(symbol, utils.today_before(1), utils.today(), interval='1m')
    detect_and_plot_signals()