import pandas as pd
import ta
import neat #pip install neat-python
import pickle

from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed

class StockTradingModel:
    def __init__(self, dataframes):
        self.dataframes = dataframes

    def preprocess_data(self, df):
        df_clean = df.loc[:, ["Close", "Low", "High"]].copy()
        df_clean["Y"] = df_clean["Close"].shift(-5)
        for i in range(1, 3):
            df_clean[f"Close_t{i}"] = df_clean["Close"].shift(i)
        df_clean["RSI"] = ta.momentum.RSIIndicator(df_clean["Close"]).rsi()
        df_clean["Williams %R"] = ta.momentum.WilliamsRIndicator(df_clean["High"], df_clean["Low"], df_clean["Close"]).williams_r()
        df_clean["ATR"] = ta.volatility.AverageTrueRange(df_clean["High"], df_clean["Low"], df_clean["Close"]).average_true_range()
        df_clean.dropna(inplace=True)
        return df_clean

    def generate_labels(self, df):
        df['y_buy'] = (df['Y'] > df['Close']).astype(int)
        df['y_sell'] = (df['Y'] < df['Close']).astype(int)
        return df

    def eval_genomes(self, genomes, config, X_train, y_train):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            predictions = [net.activate(x) for x in X_train]
            genome.fitness = -mean_squared_error(y_train, predictions)

    def run_neat(self, X_train, y_train):
        config_path = "path_to_neat_config_file"
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(5))

        winner = p.run(lambda genomes, config: self.eval_genomes(genomes, config, X_train, y_train), 50)
        with open('winner.pkl', 'wb') as f:
            pickle.dump(winner, f)

        return neat.nn.FeedForwardNetwork.create(winner, config)

    def optimize_model(self, X_train, y_train):
        neat_model = self.run_neat(X_train, y_train)
        return neat_model

    def run_optimization(self, df):
        df_clean = self.preprocess_data(df)
        df_clean = self.generate_labels(df_clean)
        X = df_clean.drop(columns=['y_buy', 'y_sell', 'Y', 'Close'])
        y_buy = df_clean['y_buy']
        y_sell = df_clean['y_sell']

        best_model_buy = self.optimize_model(X, y_buy)
        best_model_sell = self.optimize_model(X, y_sell)

        return best_model_buy, best_model_sell

    def get_results(self):
        results = Parallel(n_jobs=-1)(delayed(self.run_optimization)(df) for df in self.dataframes)
        return results

# Función para cargar DataFrames desde un archivo CSV (ajusta según sea necesario)
def get_dataframes():
    # Ejemplo: dfs = [pd.read_csv(f'file_{i}.csv') for i in range(n)]
    dfs = []
    return dfs

if __name__ == "__main__":
    dataframes = get_dataframes()
    model = StockTradingModel(dataframes)
    results = model.get_results()
    print(results)
