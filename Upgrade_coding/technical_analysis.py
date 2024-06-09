import pandas as pd
import ta
import optuna
from itertools import combinations

class TradingStrategy:
    def __init__(self, data, initial_capital=1_000_000, com=0.125 / 100):
        self.data = data
        self.initial_capital = initial_capital
        self.com = com
        self.signals = None
    
    def calculate_indicators(self):
        # Calcular el RSI (Relative Strength Index)
        self.data['RSI'] = ta.momentum.RSIIndicator(self.data['Close'], window=14).rsi()

        # Calcular el ATR (Average True Range)
        self.data['ATR'] = ta.volatility.AverageTrueRange(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            window=14
        ).average_true_range()

        # Calcular el Williams %R
        self.data['Williams_%R'] = ta.momentum.WilliamsRIndicator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            lbp=14
        ).williams_r()
    
    def generate_signals(self, indicators):
        self.signals = self.data.copy()
        self.signals['Buy_Signal'] = 0
        self.signals['Sell_Signal'] = 0
        
        if 'RSI' in indicators:
            self.signals['Buy_Signal'] |= ((self.signals['RSI'] < 30)).astype(int)
            self.signals['Sell_Signal'] |= ((self.signals['RSI'] > 70)).astype(int)
        
        if 'ATR' in indicators:
            # Definir umbral de volatilidad para señales de compra/venta
            atr_threshold = self.signals['ATR'].mean()
            self.signals['Buy_Signal'] |= ((self.signals['ATR'] > atr_threshold)).astype(int)
            self.signals['Sell_Signal'] |= ((self.signals['ATR'] < atr_threshold)).astype(int)
        
        if 'Williams_%R' in indicators:
            self.signals['Buy_Signal'] |= ((self.signals['Williams_%R'] < -80)).astype(int)
            self.signals['Sell_Signal'] |= ((self.signals['Williams_%R'] > -20)).astype(int)
        
        return self.signals
    
    def create_signals(self, strategy: str, **kwargs):
        self.signals = self.data.copy()

        rsi_1 = ta.momentum.RSIIndicator(self.signals.Close, kwargs["rsi_window"])  # RSI para compra
        rsi_2 = ta.momentum.RSIIndicator(self.signals.Close, kwargs["rsi_window"])  # RSI para venta

        bollinger = ta.volatility.BollingerBands(self.signals.Close, 
                                                 kwargs["bollinger_window"], 
                                                 kwargs["bollinger_std"])

        self.signals["rsi"] = rsi_1.rsi()
        self.signals["rsi2"] = rsi_2.rsi()

        self.signals["BUY_SIGNAL"] = (self.signals["rsi"] < kwargs["rsi_lower_threshold"]) & \
                                     (bollinger.bollinger_lband_indicator().astype(bool))
        self.signals["SELL_SIGNAL"] = (self.signals["rsi2"] > kwargs["rsi_upper_threshold"]) & \
                                      (bollinger.bollinger_hband_indicator().astype(bool))
        return self.signals.dropna()
    
    def backtest(self, trial):
        # Parámetros
        capital = self.initial_capital
        n_shares = trial.suggest_float("n_shares", 10, 150)
        stop_loss = trial.suggest_float("stop_loss", 0.05, 0.15)
        take_profit = trial.suggest_float("take_profit", 0.05, 0.15)
        max_active_operations_buy = 500
        max_active_operations_sell = 500

        rsi_window = trial.suggest_int("rsi_window", 5, 50)
        rsi_lower_threshold = trial.suggest_int("rsi_lower_threshold", 10, 30)
        rsi_upper_threshold = trial.suggest_int("rsi_upper_threshold", 60, 90)
        bollinger_window = trial.suggest_int("bollinger_window", 10, 50)
        bollinger_std = trial.suggest_float("bollinger_std", 1.0, 3.0)

        technical_data = self.create_signals(
            strategy='rsi_bollinger',
            rsi_window=rsi_window, 
            rsi_lower_threshold=rsi_lower_threshold,
            rsi_upper_threshold=rsi_upper_threshold,
            bollinger_window=bollinger_window,
            bollinger_std=bollinger_std
        )

        long_positions = []
        short_positions = []
        portfolio_value = [capital]

        for i, row in technical_data.iterrows():
            # Cerrar posiciones long que cumplan con SL o TP
            long_pos_copy = long_positions.copy()
            for pos in long_pos_copy:
                if row.Close < pos["stop_loss"]:
                    capital += row.Close * pos["n_shares"] * (1 - self.com)
                    long_positions.remove(pos)
                elif row.Close > pos["take_profit"]:
                    capital += row.Close * pos["n_shares"] * (1 - self.com)
                    long_positions.remove(pos)
                    
            # Cerrar posiciones short que cumplan con SL o TP
            short_pos_copy = short_positions.copy()
            for pos in short_pos_copy:
                if row.Close > pos["stop_loss"]:
                    capital -= row.Close * pos["n_shares"] * (1 + self.com)
                    short_positions.remove(pos)
                elif row.Close < pos["take_profit"]:
                    capital -= row.Close * pos["n_shares"] * (1 + self.com)
                    short_positions.remove(pos)

            # Verificar señal de compra
            if row.BUY_SIGNAL and len(long_positions) < max_active_operations_buy:
                if capital > row.Close * (1 + self.com) * n_shares:
                    capital -= row.Close * (1 + self.com) * n_shares
                    long_positions.append({
                        "type": "LONG",
                        "bought_at": row.Close,
                        "n_shares": n_shares,
                        "stop_loss": row.Close * (1 - stop_loss),
                        "take_profit": row.Close * (1 + take_profit)
                    })

            # Verificar señal de venta
            if row.SELL_SIGNAL and len(short_positions) < max_active_operations_sell:
                if capital > row.Close * (1 + self.com) * n_shares:
                    capital += row.Close * (1 - self.com) * n_shares
                    short_positions.append({
                        "type": "SHORT",
                        "sold_at": row.Close,
                        "n_shares": n_shares,
                        "stop_loss": row.Close * (1 + stop_loss),
                        "take_profit": row.Close * (1 - take_profit)
                    })

            # Valor del portafolio a lo largo del tiempo
            long_position_value = sum(pos["n_shares"] * row.Close for pos in long_positions)
            short_position_value = sum(pos["n_shares"] * (pos["sold_at"] - row.Close) for pos in short_positions)
            portfolio_value.append(capital + long_position_value + short_position_value)

        # Cerrar todas las posiciones al final
        long_pos_copy = long_positions.copy()
        for pos in long_pos_copy:
            capital += row.Close * pos["n_shares"] * (1 - self.com)
            long_positions.remove(pos)

        short_pos_copy = short_positions.copy()
        for pos in short_pos_copy:
            capital -= row.Close * pos["n_shares"] * (1 + self.com)
            short_positions.remove(pos)

        portfolio_value.append(capital)
        return portfolio_value[-1]

def run_optimization(data):
    strategy = TradingStrategy(data)
    strategy.calculate_indicators()
    
    indicators = ['RSI', 'ATR', 'Williams_%R']
    all_combinations = []

    for i in range(1, len(indicators) + 1):
        comb = list(combinations(indicators, i))
        all_combinations.extend(comb)

    best_results = []

    for combination in all_combinations:
        print(f"Optimizing for combination: {combination}")
        strategy.generate_signals(combination)
        study = optuna.create_study(direction='maximize')
        study.optimize(strategy.backtest, n_trials=50)
        best_results.append((combination, study.best_value))
    
    best_results.sort(key=lambda x: x[1], reverse=True)
    return best_results

# Cargar datos y ejecutar la optimización
data = pd.read_csv("./technical_analysis/data/AAPL/aapl_project_5m_train.csv").dropna()

results = run_optimization(data)

print("Best combinations and results:")
for res in results:
    print(f"Combination: {res[0]}, Result: {res[1]}")

