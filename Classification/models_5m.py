# Importing libraries....
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import ta

# Model Libraries...
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score, f1_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

from utils.utils import StockTradingModel

pd.read_csv("./technical_analysis/data/AAPL/aapl_project_5m_train.csv").dropna()
# Importing datasets to train models :)
# Train & Test
data_aapl_1m_Tr = pd.read_csv("data/AAPL/aapl_project_1m_train.csv")
data_aapl_5m_Tr = pd.read_csv("data/AAPL/aapl_project_train.csv")
data_btc_1m_Tr  = pd.read_csv("data/BTC-USD/btc_project_1m_train.csv")
data_btc_5m_Tr  = pd.read_csv("data/BTC-USD/btc_project_train.csv")

# Validation
data_aapl_1m_ts = pd.read_csv("data/AAPL/aapl_project_1m_test.csv")
data_aapl_5m_ts = pd.read_csv("data/AAPL/aapl_project_test.csv")
data_btc_1m_ts  = pd.read_csv("data/BTC-USD/btc_project_1m_test.csv")
data_btc_5m_ts  = pd.read_csv("data/BTC-USD/btc_project_test.csv")

dataframes = {
    'AAPL_5M_TRAIN': data_aapl_5m_Tr,
    'BTC_5M_TRAIN': data_btc_5m_Tr,
    'AAPL_1M_TRAIN': data_aapl_1m_Tr,
    'BTC_1M_TRAIN': data_btc_1m_Tr
}

# Instanciar y usar la clase
trading_model = StockTradingModel(dataframes)
trading_model.train_and_evaluate()
results = trading_model.get_results()

# Mostrar resultados en un DataFrame
results_df = pd.DataFrame(results)
results_df.to_html("model_results.html")

print("Results saved to 'model_results.html'")

