# Only the code would be run.....

# project_name/technical_analysis/main.py
import pandas as pd 
import optuna
from technical_analysis.utils import create_signals


# Importar dataset
data = pd.read_csv("./technical_analysis/data/AAPL/aapl_project_5m_train.csv").dropna()

def profit(trial, combination: int, **kwargs):
    capital = 1_000_000
    n_shares = trial.suggest_float("n_shares", 10, 150)
    stop_loss = trial.suggest_float("stop_loss", 0.05, 0.15)
    take_profit = trial.suggest_float("take_profit", 0.05, 0.15)
    max_active_operations_buy = 500
    max_active_operations_sell = 500
    COM = 0.125 / 100  # Comisión

    if combination & 0b001:
        rsi_window = trial.suggest_int("rsi_window", 100, 250)
        rsi_lower_threshold = trial.suggest_int("rsi_lower_threshold", 10, 30)
        rsi_upper_threshold = trial.suggest_int("rsi_upper_threshold", 60, 90)
    else:
        rsi_window = rsi_lower_threshold = rsi_upper_threshold = None

    if combination & 0b010:
        williams_window = trial.suggest_int("williams_window", 100, 250)
        williams_r_lower_threshold = trial.suggest_int("williams_r_lower_threshold", -100, -80)
        williams_r_upper_threshold = trial.suggest_int("williams_r_upper_threshold", -20, 0)
    else:
        williams_window = williams_r_lower_threshold = williams_r_upper_threshold = None

    if combination & 0b100:
        atr_window = trial.suggest_int("atr_window", 100, 250)
    else:
        atr_window = None

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
        long_pos_copy = long_positions.copy()
        for pos in long_pos_copy:
            if row.Close < pos["stop_loss"]:
                capital += row.Close * pos["n_shares"] * (1 - COM)
                long_positions.remove(pos)
            elif row.Close > pos["take_profit"]:
                capital += row.Close * pos["n_shares"] * (1 - COM)
                long_positions.remove(pos)

        short_pos_copy = short_positions.copy()
        for pos in short_pos_copy:
            if row.Close > pos["stop_loss"]:
                capital -= row.Close * pos["n_shares"] * (1 + COM)
                short_positions.remove(pos)
            elif row.Close < pos["take_profit"]:
                capital -= row.Close * pos["n_shares"] * (1 + COM)
                short_positions.remove(pos)

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

        long_position_value = sum(pos["n_shares"] * row.Close for pos in long_positions)
        short_position_value = sum(pos["n_shares"] * (pos["sold_at"] - row.Close) for pos in short_positions)
        portfolio_value.append(capital + long_position_value + short_position_value)

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

def main():
    # Diccionario para almacenar los mejores trials por combinación
    best_trials = {}

    for combination in range(8):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda x: profit(x, combination), n_trials=50, n_jobs=-1)
        best_trials[combination] = study.best_trial

    print("Best trials for each combination:")
    for combination, trial in best_trials.items():
        print(f"Combination: {combination}")
        print(f"Value: {trial.value}")
        print("Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
