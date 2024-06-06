# The central code for Project 2....

# Importing libraries...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import optuna 

# Loading data training...
aapl_1m = pd.read_csv('../data/AAPL/aapl_project_1m_train.csv')
aapl_5m = pd.read_csv('../data/AAPL/aapl_project_5m_train.csv')
btc_1h = pd.read_csv('../data/BTC-USD/btc_project_1m_train.csv')
btc_1d = pd.read_csv('../data/BTC-USD/btc_project_5m_train.csv')

# 

# Importing dataset
data = pd.read_csv("aapl_project_train.csv").dropna()

# Pre-calculating indicators to avoid repeated computation
def precalculate_indicators(data, rsi_window, williams_window, atr_window):
    data = data.copy()
    data["RSI"] = ta.momentum.RSIIndicator(data.Close, rsi_window).rsi()
    data["Williams %R"] = ta.momentum.WilliamsRIndicator(data.High, data.Low, data.Close, williams_window).williams_r()
    data["ATR"] = ta.volatility.AverageTrueRange(data.High, data.Low, data.Close, atr_window).average_true_range()
    return data.dropna()

# Function to create signals buys and sell
def create_signals(data: pd.DataFrame, combination: int, **kwargs):
    data = data.copy()

    # Apply combinations
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

# Backtesting function and profit def
def profit(trial, combination: int, **kwargs):
    # Parameters
    capital = 1_000_000
    n_shares = trial.suggest_float("n_shares", 10, 150)
    stop_loss = trial.suggest_float("stop_loss", 0.05, 0.15)
    take_profit = trial.suggest_float("take_profit", 0.05, 0.15)
    max_active_operations_buy = 500
    max_active_operations_sell = 500
    COM = 0.125 / 100  # Comisión

    if combination & 0b001:
        rsi_window = trial.suggest_int("rsi_window", 5, 50)
        rsi_lower_threshold = trial.suggest_int("rsi_lower_threshold", 10, 30)
        rsi_upper_threshold = trial.suggest_int("rsi_upper_threshold", 60, 90)
    else:
        rsi_window = rsi_lower_threshold = rsi_upper_threshold = None

    if combination & 0b010:
        williams_window = trial.suggest_int("williams_window", 10, 50)
        williams_r_lower_threshold = trial.suggest_int("williams_r_lower_threshold", -100, -80)
        williams_r_upper_threshold = trial.suggest_int("williams_r_upper_threshold", -20, 0)
    else:
        williams_window = williams_r_lower_threshold = williams_r_upper_threshold = None

    if combination & 0b100:
        atr_window = trial.suggest_int("atr_window", 10, 50)
    else:
        atr_window = None

    # Precalculate indicators
    technical_data = create_signals(data,
                                    combination=combination,
                                    rsi_window=rsi_window,
                                    rsi_lower_threshold=rsi_lower_threshold,
                                    rsi_upper_threshold=rsi_upper_threshold,
                                    williams_window=williams_window,
                                    williams_r_lower_threshold=williams_r_lower_threshold,
                                    williams_r_upper_threshold=williams_r_upper_threshold,
                                    atr_window=atr_window)

    long_positions = []
    short_positions = []
    portfolio_value = [capital]

    for i, row in technical_data.iterrows():
        # Close positions long they have SL Or TP
        long_pos_copy = long_positions.copy()
        for pos in long_pos_copy:
            if row.Close < pos["stop_loss"]:
                capital += row.Close * pos["n_shares"] * (1 - COM)
                long_positions.remove(pos)
            elif row.Close > pos["take_profit"]:
                capital += row.Close * pos["n_shares"] * (1 - COM)
                long_positions.remove(pos)

        # Close positions short they have SL Or TP
        short_pos_copy = short_positions.copy()
        for pos in short_pos_copy:
            if row.Close > pos["stop_loss"]:
                capital -= row.Close * pos["n_shares"] * (1 + COM)
                short_positions.remove(pos)
            elif row.Close < pos["take_profit"]:
                capital -= row.Close * pos["n_shares"] * (1 + COM)
                short_positions.remove(pos)

        # Check BUY SIGNAL
        if row.BUY_SIGNAL and len(long_positions) < max_active_operations_buy:
            if capital > row.Close * (1 + COM) * n_shares:
                capital -= row.Close * (1 + COM) * n_shares
                long_positions.append({
                    "type": "LONG",
                    "bought_at": row.Close,
                    "n_shares": n_shares,
                    "stop_loss": row.Close * (1 - stop_loss),
                    "take_profit": row.Close * (1 + take_profit)
                })

        # Check SELL SIGNAL
        if row.SELL_SIGNAL and len(short_positions) < max_active_operations_sell:
            if capital > row.Close * (1 + COM) * n_shares:
                capital += row.Close * (1 - COM) * n_shares
                short_positions.append({
                    "type": "SHORT",
                    "sold_at": row.Close,
                    "n_shares": n_shares,
                    "stop_loss": row.Close * (1 + stop_loss),
                    "take_profit": row.Close * (1 - take_profit)
                })

        # Portfolio value long to time (portfolio is longs and short positions)
        long_position_value = sum(pos["n_shares"] * row.Close for pos in long_positions)
        short_position_value = sum(pos["n_shares"] * (pos["sold_at"] - row.Close) for pos in short_positions)
        portfolio_value.append(capital + long_position_value + short_position_value)

    # Close all positions
    long_pos_copy = long_positions.copy()
    for pos in long_pos_copy:
        capital += row.Close * pos["n_shares"] * (1 - COM)
        long_positions.remove(pos)

    short_pos_copy = short_positions.copy()
    for pos in short_pos_copy:
        capital -= row.Close * pos["n_shares"] * (1 + COM)
        short_positions.remove(pos)

    portfolio_value.append(capital)
    return portfolio_value[-1]

# Optimize
study = optuna.create_study(direction="maximize")
study.optimize(lambda x: profit(x, 3), n_trials=50, n_jobs=-1)  # Use parallel execution # combination 3

print("Best trial:")
trial = study.best_trial
print(f"Value: {trial.value}")
print("Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
