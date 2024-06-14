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
from sklearn.ensemble import VotingClassifier

# Importing the StockTradingModel class from utils
from utils.utils import StockTradingModel

# Importing datasets to train models :)
# Train & Test
data_aapl_1m_Tr = pd.read_csv("./technical_analysis/data/AAPL/aapl_project_1m_train.csv").dropna()
data_aapl_5m_Tr = pd.read_csv("./technical_analysis/data/AAPL/aapl_project_5m_train.csv").dropna()
data_btc_1m_Tr  = pd.read_csv("./technical_analysis/data/BTC-USD/btc_project_1m_train.csv").dropna()
data_btc_5m_Tr  = pd.read_csv("./technical_analysis/data/BTC-USD/btc_project_5m_train.csv").dropna()

# Validation
data_aapl_1m_ts = pd.read_csv("./technical_analysis/data/AAPL/aapl_project_1m_test.csv").dropna()
data_aapl_5m_ts = pd.read_csv("./technical_analysis/data/AAPL/aapl_project_5m_test.csv").dropna()
data_btc_1m_ts  = pd.read_csv("./technical_analysis/data/BTC-USD/btc_project_1m_test.csv").dropna()
data_btc_5m_ts  = pd.read_csv("./technical_analysis/data/BTC-USD/btc_project_5m_test.csv").dropna()

# Organizing data into a dictionary
dataframes = {
    'AAPL_5M_TRAIN': data_aapl_5m_Tr,
    'BTC_5M_TRAIN': data_btc_5m_Tr,
    'AAPL_1M_TRAIN': data_aapl_1m_Tr,
    'BTC_1M_TRAIN': data_btc_1m_Tr
}

# Instantiate the model
model = StockTradingModel(list(dataframes.values()))

# Run optimization and training
results = model.get_results()

# Display results
for i, (df_name, (buy_model, sell_model)) in enumerate(zip(dataframes.keys(), results)):
    print(f"Results for {df_name}:")
    print("Buy Model Details:")
    print(buy_model)
    print("Sell Model Details:")
    print(sell_model)
    print()
