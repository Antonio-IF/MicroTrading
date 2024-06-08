# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import ta

# Load the dataset
btc_5m = pd.read_csv("./technical_analysis/data/BTC-USD/btc_project_5m_test.csv").dropna()

# Optimal parameters from Optuna
best_params = {
    "n_shares": 107.76747742408709,
    "stop_loss": 0.0784524647519304,
    "take_profit": 0.08443871610882869,
    "rsi_window": 100,
    "rsi_lower_threshold": 17,
    "rsi_upper_threshold": 84
}

# Calculate the RSI using the optimal parameters
rsi_btc_5m = ta.momentum.RSIIndicator(
    close=btc_5m['Close'], window=best_params["rsi_window"]
)
btc_5m['RSI'] = rsi_btc_5m.rsi()

# Create DataFrame for signals
technical_data_btc_5m = pd.DataFrame()
technical_data_btc_5m["Close"] = btc_5m["Close"]
technical_data_btc_5m["RSI"] = btc_5m["RSI"]
technical_data_btc_5m = technical_data_btc_5m.dropna()

# Generate buy and sell signals using the optimal thresholds
technical_data_btc_5m["BUY_SIGNAL"] = technical_data_btc_5m["RSI"] < best_params["rsi_lower_threshold"]
technical_data_btc_5m["SELL_SIGNAL"] = technical_data_btc_5m["RSI"] > best_params["rsi_upper_threshold"]

# Backtesting parameters
capital = 1_000_000
n_shares = best_params["n_shares"]
stop_loss = best_params["stop_loss"]
take_profit = best_params["take_profit"]
COM = 0.125 / 100  # Commission

# Initialize
long_positions = []
short_positions = []
portfolio_value = [capital]

# Backtesting
for i, row in technical_data_btc_5m.iterrows():
    # Close positions that have reached take profit or stop loss
    long_pos_copy = long_positions.copy()
    for pos in long_pos_copy: 
        if row.Close < pos["stop_loss"]:
            # Loss
            capital += row.Close * pos["n_shares"] * (1 - COM)
            long_positions.remove(pos)
        elif row.Close > pos["take_profit"]:
            # Gain
            capital += row.Close * pos["n_shares"] * (1 - COM)
            long_positions.remove(pos)
            
    short_pos_copy = short_positions.copy()
    for pos in short_pos_copy:
        if row.Close > pos["stop_loss"]:
            # Loss
            capital -= row.Close * pos["n_shares"] * (1 + COM)
            short_positions.remove(pos)
        elif row.Close < pos["take_profit"]:
            # Gain
            capital -= row.Close * pos["n_shares"] * (1 + COM)
            short_positions.remove(pos)

    # Check buy signal
    if row.BUY_SIGNAL:
        if capital > row.Close * (1 + COM) * n_shares: 
            capital -= row.Close * (1 + COM) * n_shares
            long_positions.append({
                "type": "LONG",
                "bought_at": row.Close,
                "n_shares": n_shares,
                "stop_loss": row.Close * (1 - stop_loss),
                "take_profit": row.Close * (1 + take_profit)
            })

    # Check sell signal
    if row.SELL_SIGNAL:
        if capital > row.Close * (1 + COM) * n_shares: 
            capital += row.Close * (1 - COM) * n_shares
            short_positions.append({
                "type": "SHORT",
                "sold_at": row.Close,
                "n_shares": n_shares,
                "stop_loss": row.Close * (1 + stop_loss),
                "take_profit": row.Close * (1 - take_profit)
            })
    
    # Portfolio value over time
    long_position_value = sum(pos["n_shares"] * row.Close for pos in long_positions)
    short_position_value = sum(pos["n_shares"] * (pos["sold_at"] - row.Close) for pos in short_positions)
    portfolio_value.append(capital + long_position_value + short_position_value)

# Close all positions
for pos in long_positions.copy():
    capital += technical_data_btc_5m.iloc[-1].Close * pos["n_shares"] * (1 - COM)
    long_positions.remove(pos)

for pos in short_positions.copy():
    capital -= technical_data_btc_5m.iloc[-1].Close * pos["n_shares"] * (1 + COM)
    short_positions.remove(pos)

portfolio_value.append(capital)

# Benchmark portfolio
capital_benchmark = 1_000_000
shares_to_buy = capital_benchmark // (technical_data_btc_5m.Close.values[0] * (1 + COM))
capital_benchmark -= shares_to_buy * technical_data_btc_5m.Close.values[0] * (1 + COM)
portfolio_value_benchmark = (shares_to_buy * technical_data_btc_5m.Close) + capital_benchmark

# Plot portfolio value
plt.plot(portfolio_value, label='Active')
plt.plot(portfolio_value_benchmark, label='Passive')
plt.title(f'Active={(portfolio_value[-1] / 1_000_000 - 1) * 100:.2f}%\n' + 
          f'Passive={(portfolio_value_benchmark.values[-1] / 1_000_000 - 1) * 100:.2f}%')
plt.xlabel('Time')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()
