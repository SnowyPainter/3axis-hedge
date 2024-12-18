import sys, os
sys.path.append('../')

import utils
import charpoon_model
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.cluster import DBSCAN

def detect_and_plot_signals():
    """
    V자 및 역V자 신호를 감지하고 차트 위에 시각화, 
    신호 군집화로 평균 매수/매도 단가를 표시.
    """
    v_signals = []
    inverted_v_signals = []
    prices = data[f"{symbol}_Price"].values
    bar = 0
    
    # 신호 감지
    while bar < len(data) - charpoon_model.seqlen:
        if bar > charpoon_model.seqlen * 2:
            start = bar - charpoon_model.seqlen - 49  # must be 49.
            end = bar
            pred = charpoon_model.predict(model, data.iloc[start:end], symbol)
            if np.argmax(pred) == 1:  # 역V자 패턴
                inverted_v_signals.append(bar)
            elif np.argmax(pred) == 2:  # V자 패턴
                v_signals.append(bar)
        
        bar += 1
    
    # 신호 군집화
    def cluster_signals(signal_indices):
        """신호를 군집화하여 각 군집의 중심(평균 단가)을 반환"""
        if not signal_indices:
            return []
        
        # 신호 위치를 DBSCAN으로 군집화
        clustering = DBSCAN(eps=10, min_samples=2).fit(np.array(signal_indices).reshape(-1, 1))
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label == -1:
                continue  # Noise 제거
            clusters.setdefault(label, []).append(signal_indices[idx])
        
        # 각 군집의 평균 단가 계산
        avg_prices = [
            (np.mean(cluster), np.mean(prices[cluster])) for cluster in clusters.values()
        ]
        return avg_prices
    
    v_clusters = cluster_signals(v_signals)
    inverted_v_clusters = cluster_signals(inverted_v_signals)
    
    # 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label=f"{symbol} Price", color="blue")
    
    # V자 및 역V자 신호
    plt.scatter(v_signals, prices[v_signals], color="green", label="V Pattern", marker="^", alpha=1, s=70)
    plt.scatter(inverted_v_signals, prices[inverted_v_signals], color="red", label="Inverted V Pattern", marker="v", alpha=1, s=70)
    
    # 평균 매수/매도 단가 점선
    for avg_bar, avg_price in v_clusters:
        plt.axhline(y=avg_price, color="green", linestyle="--", alpha=0.7, label="Avg Buy Price")
    for avg_bar, avg_price in inverted_v_clusters:
        plt.axhline(y=avg_price, color="red", linestyle="--", alpha=0.7, label="Avg Sell Price")
    
    # 기타 설정
    plt.title(f"Price Chart with V and Inverted V Patterns: {symbol}")
    plt.xlabel("Time (Bar)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{symbol}_v_inverted_v_patterns.png")
    plt.show()

for symbol in ["DOGE-USD", "BTC-USD", "XRP-USD"]:
    symbols = [symbol]
    data = utils.load_historical_for_learning(symbol, utils.today_before(30), utils.today(), interval='1h')
    if os.path.exists(f"best_CHARPOON_{symbol}_finetuned_model.h5"):
        print(f"Model for {symbol} already exists. Skipping training.")
        model = load_model(f"best_CHARPOON_{symbol}_finetuned_model.h5")
    else:
        model = charpoon_model.finetune_model(symbol, charpoon_model.calculate_technical_indicators(data, symbol))
    
    detect_and_plot_signals()