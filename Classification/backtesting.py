import pandas as pd
import matplotlib.pyplot as plt
import ta
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np

class TradingStrategyWithModels:
    def __init__(self, data, buy_model, sell_model, initial_capital, com=0.1 / 100):
        self.data = data
        self.buy_model = buy_model
        self.sell_model = sell_model
        self.initial_capital = initial_capital
        self.com = com
        self.signals = None
    
    def calculate_indicators(self):
        self.data['RSI'] = ta.momentum.RSIIndicator(self.data['Close'], window=14).rsi()
        self.data['ATR'] = ta.volatility.AverageTrueRange(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            window=14
        ).average_true_range()
        self.data['Williams_%R'] = ta.momentum.WilliamsRIndicator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            lbp=14
        ).williams_r()
        self.data.dropna(inplace=True)
    
    def generate_signals(self):
        self.signals = self.data.copy()
        self.signals['Buy_Signal'] = self.buy_model.predict(self.signals[['RSI', 'ATR', 'Williams_%R']])
        self.signals['Sell_Signal'] = self.sell_model.predict(self.signals[['RSI', 'ATR', 'Williams_%R']])
        return self.signals
    
    def backtest(self):
        capital = self.initial_capital
        n_shares = 100  # Ajuste en número de acciones
        stop_loss = 0.05  # Ajuste de stop loss
        take_profit = 0.10  # Ajuste de take profit
        max_active_operations_buy = 50  # Ajuste del número máximo de operaciones activas de compra
        max_active_operations_sell = 50  # Ajuste del número máximo de operaciones activas de venta

        long_positions = []
        short_positions = []
        portfolio_value = [capital]

        for i, row in self.signals.iterrows():
            long_pos_copy = long_positions.copy()
            for pos in long_pos_copy:
                if row.Close < pos["stop_loss"]:
                    capital += row.Close * pos["n_shares"] * (1 - self.com)
                    long_positions.remove(pos)
                elif row.Close > pos["take_profit"]:
                    capital += row.Close * pos["n_shares"] * (1 - self.com)
                    long_positions.remove(pos)
                    
            short_pos_copy = short_positions.copy()
            for pos in short_pos_copy:
                if row.Close > pos["stop_loss"]:
                    capital -= row.Close * pos["n_shares"] * (1 + self.com)
                    short_positions.remove(pos)
                elif row.Close < pos["take_profit"]:
                    capital -= row.Close * pos["n_shares"] * (1 + self.com)
                    short_positions.remove(pos)

            if row.Buy_Signal and len(long_positions) < max_active_operations_buy:
                if capital > row.Close * (1 + self.com) * n_shares:
                    capital -= row.Close * (1 + self.com) * n_shares
                    long_positions.append({
                        "type": "LONG",
                        "bought_at": row.Close,
                        "n_shares": n_shares,
                        "stop_loss": row.Close * (1 - stop_loss),
                        "take_profit": row.Close * (1 + take_profit)
                    })

            if row.Sell_Signal and len(short_positions) < max_active_operations_sell:
                if capital > row.Close * (1 + self.com) * n_shares:
                    capital += row.Close * (1 - self.com) * n_shares
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
            capital += self.signals.iloc[-1].Close * pos["n_shares"] * (1 - self.com)
            long_positions.remove(pos)

        short_pos_copy = short_positions.copy()
        for pos in short_pos_copy:
            capital -= self.signals.iloc[-1].Close * pos["n_shares"] * (1 + self.com)
            short_positions.remove(pos)

        portfolio_value.append(capital)
        return portfolio_value

def run_backtest_with_models(data, buy_model, sell_model, initial_capital):
    strategy = TradingStrategyWithModels(data, buy_model, sell_model, initial_capital)
    strategy.calculate_indicators()
    
    strategy.generate_signals()
    portfolio_value = strategy.backtest()
    
    return portfolio_value

def calculate_benchmark(data, capital_benchmark):
    shares_to_buy = capital_benchmark // (data['Close'].values[0] * (1 + 0.00125))
    capital_benchmark -= shares_to_buy * data['Close'].values[0] * (1 + 0.00125)
    portfolio_value_benchmark = (shares_to_buy * data['Close']) + capital_benchmark
    return portfolio_value_benchmark

def plot_portfolio_value(portfolio_value, portfolio_value_benchmark, title, filename):
    plt.figure()
    plt.plot(portfolio_value, label='Active')
    plt.plot(portfolio_value_benchmark, label='Passive')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def prepare_data_and_models():
    # Cargar datos de prueba
    data_aapl_1m_ts = pd.read_csv("./Classification/data/AAPL/aapl_project_1m_test.csv").dropna()
    data_aapl_5m_ts = pd.read_csv("./Classification/data/AAPL/aapl_project_5m_test.csv").dropna()
    # data_btc_1m_ts  = pd.read_csv("./Classification/data/BTC-USD/btc_project_1m_test.csv").dropna()
    # data_btc_5m_ts  = pd.read_csv("./Classification/data/BTC-USD/btc_project_5m_test.csv").dropna()

    # Calcular los indicadores para los datos de prueba
    for df in [data_aapl_1m_ts, data_aapl_5m_ts]:
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['ATR'] = ta.volatility.AverageTrueRange(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=14
        ).average_true_range()
        df['Williams_%R'] = ta.momentum.WilliamsRIndicator(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            lbp=14
        ).williams_r()
        df.dropna(inplace=True)

    # Definir los mejores modelos para compra y venta

    # AAPL 1M
    buy_model_aapl_1m = VotingClassifier(estimators=[
        ('LogisticRegression', LogisticRegression(C=0.06265349853986023, l1_ratio=0.3436453147648226, max_iter=30000)),
        ('SVC', SVC(C=4.654173314213181, gamma='auto')),
        ('XGB', XGBClassifier(base_score=None, booster='gbtree', learning_rate=0.03814684286747783, max_depth=10, max_leaves=0, n_estimators=178))
    ])

    sell_model_aapl_1m = VotingClassifier(estimators=[
        ('LogisticRegression', LogisticRegression(C=3.1061278475426617, l1_ratio=0.6628663149844953, max_iter=30000)),
        ('SVC', SVC(C=0.017847466414163057, gamma='auto', kernel='linear')),
        ('XGB', XGBClassifier(base_score=None, booster='gbtree', learning_rate=0.1147284945320656, max_depth=10, max_leaves=0, n_estimators=115))
    ])

    # AAPL 5M
    buy_model_aapl_5m = VotingClassifier(estimators=[
        ('LogisticRegression', LogisticRegression(C=0.0016672071354947374, l1_ratio=0.9299944673622269, max_iter=30000)),
        ('SVC', SVC(C=990.6740326830704, gamma='auto')),
        ('XGB', XGBClassifier(base_score=None, booster='gblinear', learning_rate=0.16834016882459785, max_depth=8, max_leaves=13, n_estimators=182))
    ])

    sell_model_aapl_5m = VotingClassifier(estimators=[
        ('LogisticRegression', LogisticRegression(C=22.306621719180473, fit_intercept=False, l1_ratio=0.6179617851107659, max_iter=30000)),
        ('SVC', SVC(C=993.9093797420461, gamma='auto')),
        ('XGB', XGBClassifier(base_score=None, booster='gbtree', learning_rate=0.1997205014266997, max_depth=7, max_leaves=14, n_estimators=200))
    ])

    # BTC 1M
    buy_model_btc_1m = VotingClassifier(estimators=[
        ('LogisticRegression', LogisticRegression(C=0.0010388294623781195, l1_ratio=0.9935625741216142, max_iter=30000)),
        ('SVC', SVC(C=0.36018202375812247, gamma='auto')),
        ('XGB', XGBClassifier(base_score=None, booster='gblinear', learning_rate=0.05417024838836022, max_depth=6, max_leaves=20, n_estimators=82))
    ])

    sell_model_btc_1m = VotingClassifier(estimators=[
        ('LogisticRegression', LogisticRegression(C=3.119632823574036, l1_ratio=0.3480188281364231, max_iter=30000)),
        ('SVC', SVC(C=0.8805855400529635, gamma='auto')),
        ('XGB', XGBClassifier(base_score=None, booster='gbtree', learning_rate=0.19873452500806038, max_depth=6, max_leaves=19, n_estimators=186))
    ])

    # BTC 5M
    buy_model_btc_5m = VotingClassifier(estimators=[
        ('LogisticRegression', LogisticRegression(C=0.0010143984975422402, fit_intercept=False, l1_ratio=0.7671361967597897, max_iter=30000)),
        ('SVC', SVC(C=0.9611592788124548, gamma='auto')),
        ('XGB', XGBClassifier(base_score=None, booster='gblinear', learning_rate=0.01917548361756064, max_depth=10, max_leaves=6, n_estimators=51))
    ])

    sell_model_btc_5m = VotingClassifier(estimators=[
        ('LogisticRegression', LogisticRegression(C=2.184421228408166, fit_intercept=False, l1_ratio=0.8524529692469743, max_iter=30000)),
        ('SVC', SVC(C=0.6179334609827659, gamma='auto')),
        ('XGB', XGBClassifier(base_score=None, booster='dart', learning_rate=0.1762776732287373, max_depth=9, max_leaves=20, n_estimators=135))
    ])

    # Entrenar los modelos con los datos de prueba
    for model in [buy_model_aapl_1m, sell_model_aapl_1m, buy_model_aapl_5m, sell_model_aapl_5m]:
        model.fit(data_aapl_5m_ts[['RSI', 'ATR', 'Williams_%R']], np.random.randint(0, 2, size=len(data_aapl_5m_ts)))

    return (data_aapl_1m_ts, data_aapl_5m_ts,
            buy_model_aapl_1m, sell_model_aapl_1m, buy_model_aapl_5m, sell_model_aapl_5m)

def main():
    (data_aapl_1m_ts, data_aapl_5m_ts,
     buy_model_aapl_1m, sell_model_aapl_1m, buy_model_aapl_5m, sell_model_aapl_5m) = prepare_data_and_models()

    # Ejecutar backtesting
    portfolio_value_aapl_1m = run_backtest_with_models(data_aapl_1m_ts, buy_model_aapl_1m, sell_model_aapl_1m, 1_000_000)
    portfolio_value_aapl_5m = run_backtest_with_models(data_aapl_5m_ts, buy_model_aapl_5m, sell_model_aapl_5m, 1_000_000)
    # portfolio_value_btc_1m = run_backtest_with_models(data_btc_1m_ts, buy_model_btc_1m, sell_model_btc_1m, 7_500_000)
    # portfolio_value_btc_5m = run_backtest_with_models(data_btc_5m_ts, buy_model_btc_5m, sell_model_btc_5m, 7_500_000)

    # Benchmark portfolio
    capital_benchmark_aapl = 1_000_000
    # capital_benchmark_btc = 5_500_000

    portfolio_value_benchmark_aapl_1m = calculate_benchmark(data_aapl_1m_ts, capital_benchmark_aapl)
    portfolio_value_benchmark_aapl_5m = calculate_benchmark(data_aapl_5m_ts, capital_benchmark_aapl)
    # portfolio_value_benchmark_btc_1m = calculate_benchmark(data_btc_1m_ts, capital_benchmark_btc)
    # portfolio_value_benchmark_btc_5m = calculate_benchmark(data_btc_5m_ts, capital_benchmark_btc)

    # Generar y guardar gráficos
    plot_portfolio_value(portfolio_value_aapl_1m, portfolio_value_benchmark_aapl_1m, 
                         f'AAPL 1M Active={(portfolio_value_aapl_1m[-1] / 1_000_000 - 1) * 100:.2f}%\n' + 
                         f'AAPL 1M Passive={(portfolio_value_benchmark_aapl_1m.values[-1] / 1_000_000 - 1) * 100:.2f}%', 
                         "aapl_1m.png")

    plot_portfolio_value(portfolio_value_aapl_5m, portfolio_value_benchmark_aapl_5m, 
                         f'AAPL 5M Active={(portfolio_value_aapl_5m[-1] / 1_000_000 - 1) * 100:.2f}%\n' + 
                         f'AAPL 5M Passive={(portfolio_value_benchmark_aapl_5m.values[-1] / 1_000_000 - 1) * 100:.2f}%', 
                         "aapl_5m.png")

    # plot_portfolio_value(portfolio_value_btc_1m, portfolio_value_benchmark_btc_1m, 
    #                      f'BTC 1M Active={(portfolio_value_btc_1m[-1] / 5_500_000 - 1) * 100:.2f}%\n' + 
    #                      f'BTC 1M Passive={(portfolio_value_benchmark_btc_1m.values[-1] / 5_500_000 - 1) * 100:.2f}%', 
    #                      "btc_1m.png")

    # plot_portfolio_value(portfolio_value_btc_5m, portfolio_value_benchmark_btc_5m, 
    #                      f'BTC 5M Active={(portfolio_value_btc_5m[-1] / 5_500_000 - 1) * 100:.2f}%\n' + 
    #                      f'BTC 5M Passive={(portfolio_value_benchmark_btc_5m.values[-1] / 5_500_000 - 1) * 100:.2f}%', 
    #                      "btc_5m.png")

if __name__ == "__main__":
    main()
