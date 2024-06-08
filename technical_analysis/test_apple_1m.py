# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import ta.momentum

# Datasets
aapl_1m = pd.read_csv("./technical_analysis/data/AAPL/aapl_project_1m_test.csv").dropna()
# aapl_5m = pd.read_csv("./technical_analysis/data/AAPL/aapl_project_5m_test.csv").dropna()
# btc_1m = pd.read_csv("./technical_analysis/data/BTC-USD/btc_project_1m_test.csv").dropna()
# btc_5m = pd.read_csv("./technical_analysis/data/BTC-USD/btc_project_5m_test.csv").dropna()

# Indicators
rsi_aapl_1m = ta.momentum.RSIIndicator(close=aapl_1m['Close'], window=150) # Optuna say 150 but a lower window have more operations a long of the time 
william_aapl_1m = ta.momentum.WilliamsRIndicator(high=aapl_1m['High'], low=aapl_1m['Low'], close=aapl_1m['Close'])

#---------------------------------------------------------#

# Dataframe for AAPL_1M
technical_data_aapl_1m = pd.DataFrame()
technical_data_aapl_1m["Close"] = aapl_1m["Close"]
technical_data_aapl_1m["RSI"] = rsi_aapl_1m.rsi()
technical_data_aapl_1m["William_R"] = william_aapl_1m.williams_r()
technical_data_aapl_1m = technical_data_aapl_1m.dropna()

#---------------------------------------------------------#

# GENERATE BUY AND SELL SIGNALS

technical_data_aapl_1m["BUY_SIGNAL"] = (technical_data_aapl_1m.RSI < 30) & (technical_data_aapl_1m.William_R < -88) 
technical_data_aapl_1m["SELL_SIGNAL"] = (technical_data_aapl_1m.RSI > 83) & (technical_data_aapl_1m.William_R > -8) 


            # Filter and print rows where BUY_SIGNAL or SELL_SIGNAL is True
            #signals = technical_data_aapl_1m[(technical_data_aapl_1m["BUY_SIGNAL"] == True) | (technical_data_aapl_1m["SELL_SIGNAL"] == True)]
            #print(signals)

# Backtesting 

#  n_shares: 149.81454328396893
#  stop_loss: 0.09449106039009689
#  take_profit: 0.050100343631403574
#  rsi_window: 150
#  rsi_lower_threshold: 30
#  rsi_upper_threshold: 83
#  williams_r_lower_threshold: -88
#  williams_r_upper_threshold: -8


# Parameters
capital = 1_000_000
n_shares = 149.81454328396893
stop_loss = 0.09449106039009689
take_profit = 0.050100343631403574
COM = 0.125 / 100  # Commission

# Initialize
long_positions = []
short_positions = []
portfolio_value = [capital]

for i, row in technical_data_aapl_1m.iterrows():
    # Close all positions that are above/under tp or sl
    long_pos_copy = long_positions.copy()
    for pos in long_pos_copy: 
        if row.Close < pos["stop_loss"]:
            # LOSS
            capital += row.Close * pos["n_shares"] * (1 - COM)
            long_positions.remove(pos)
        elif row.Close > pos["take_profit"]:
            # PROFIT
            capital += row.Close * pos["n_shares"] * (1 - COM)
            long_positions.remove(pos)
            
    short_pos_copy = short_positions.copy()
    for pos in short_pos_copy:
        if row.Close > pos["stop_loss"]:
            # LOSS
            capital -= row.Close * pos["n_shares"] * (1 + COM)
            short_positions.remove(pos)
        elif row.Close < pos["take_profit"]:
            # PROFIT
            capital -= row.Close * pos["n_shares"] * (1 + COM)
            short_positions.remove(pos)

    # Check if trading signal is True for buy
    if row.BUY_SIGNAL:
        # Check if we have enough cash
        if capital > row.Close * (1 + COM) * n_shares: 
            capital -= row.Close * (1 + COM) * n_shares
            long_positions.append({
                "type": "LONG",
                "bought_at": row.Close,
                "n_shares": n_shares,
                "stop_loss": row.Close * (1 - stop_loss),
                "take_profit": row.Close * (1 + take_profit)
            })

    # Check if trading signal is True for sell
    if row.SELL_SIGNAL:
        # Check if we have enough cash
        if capital > row.Close * (1 + COM) * n_shares: 
            capital += row.Close * (1 - COM) * n_shares
            short_positions.append({
                "type": "SHORT",
                "sold_at": row.Close,
                "n_shares": n_shares,
                "stop_loss": row.Close * (1 + stop_loss),
                "take_profit": row.Close * (1 - take_profit)
            })
    
    # Portfolio value through time
    long_position_value = sum(pos["n_shares"] * row.Close for pos in long_positions)
    short_position_value = sum(pos["n_shares"] * (pos["sold_at"] - 
                                                  row.Close) for pos in short_positions)
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

# Benchmark portfolio
capital_benchmark = 1_000_000
shares_to_buy = capital_benchmark // (technical_data_aapl_1m.Close.values[0] * (1 + COM))
capital_benchmark -= shares_to_buy * technical_data_aapl_1m.Close.values[0] * (1 + COM)
portfolio_value_benchmark = (shares_to_buy * technical_data_aapl_1m.Close) + capital_benchmark

# Plot the portfolio value
plt.plot(portfolio_value, label='Active')
plt.plot(portfolio_value_benchmark, label='Passive')
plt.title(f'Active={(portfolio_value[-1] / 1_000_000 - 1) * 100:.2f}%\n' + 
          f'Passive={(portfolio_value_benchmark.values[-1] / 1_000_000 - 1) * 100:.2f}%')
plt.xlabel('Time')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()



#---------------------------------------------------------#



