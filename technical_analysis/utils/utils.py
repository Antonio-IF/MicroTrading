# project_name/utils/utils.py

import pandas as pd 
import ta 

def precalculate_indicators(data, rsi_window, williams_window, atr_window):
    data = data.copy()
    data["RSI"] = ta.momentum.RSIIndicator(data.Close, rsi_window).rsi()
    data["Williams %R"] = ta.momentum.WilliamsRIndicator(data.High, data.Low, data.Close, williams_window).williams_r()
    data["ATR"] = ta.volatility.AverageTrueRange(data.High, data.Low, data.Close, atr_window).average_true_range()
    return data.dropna()

def create_signals(data: pd.DataFrame, combination: int, **kwargs):
    data = data.copy()

    if combination & 0b001:
        data["RSI"] = ta.momentum.RSIIndicator(data.Close, kwargs["rsi_window"]).rsi()
        data["RSI_SIGNAL"] = (data["RSI"] < kwargs["rsi_lower_threshold"]) | (data["RSI"] > kwargs["rsi_upper_threshold"])
    else:
        data["RSI_SIGNAL"] = True

    if combination & 0b010:
        data["Williams %R"] = ta.momentum.WilliamsRIndicator(data.High, data.Low, data.Close, kwargs["williams_window"]).williams_r()
        data["WILLIAMS_SIGNAL"] = (data["Williams %R"] < kwargs["williams_r_lower_threshold"]) | (data["Williams %R"] > kwargs["williams_r_upper_threshold"])
    else:
        data["WILLIAMS_SIGNAL"] = True

    if combination & 0b100:
        data["ATR"] = ta.volatility.AverageTrueRange(data.High, data.Low, data.Close, kwargs["atr_window"]).average_true_range()
        data["ATR_SIGNAL"] = (data["ATR"] > data["ATR"].rolling(window=kwargs["atr_window"]).mean())
    else:
        data["ATR_SIGNAL"] = True

    data["BUY_SIGNAL"] = data["RSI_SIGNAL"] & data["WILLIAMS_SIGNAL"] & data["ATR_SIGNAL"]
    data["SELL_SIGNAL"] = ~data["BUY_SIGNAL"]

    return data.dropna()
