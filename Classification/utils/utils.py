# Importing libraries....
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import ta

# Model Libraries...
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from joblib import Parallel, delayed

class StockTradingModel:
    def __init__(self, dataframes):
        self.dataframes = dataframes
        self.models = {
            'LogisticRegression': LogisticRegression,
            'SVC': SVC,
            'XGB': XGBClassifier
        }
        self.results = {}

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

    def objective(self, trial, model_name, X_train, y_train):
        if model_name == 'LogisticRegression':
            C = trial.suggest_float('C', 1e-3, 1e3, log=True)
            fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
            l1_ratio = trial.suggest_float('l1_ratio', 0, 1)
            model = LogisticRegression(C=C, fit_intercept=fit_intercept, penalty='elasticnet', l1_ratio=l1_ratio, solver='saga', max_iter=30000)
        elif model_name == 'SVC':
            C = trial.suggest_float('C', 1e-3, 1e3, log=True)
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            model = SVC(C=C, kernel=kernel, gamma=gamma)
        elif model_name == 'XGB':
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 2, 10)
            max_leaves = trial.suggest_int('max_leaves', 0, 20)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
            booster = trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart'])
            gamma = trial.suggest_float('gamma', 0, 5)
            reg_alpha = trial.suggest_float('reg_alpha', 0, 5)
            reg_lambda = trial.suggest_float('reg_lambda', 0, 5)
            model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, max_leaves=max_leaves,
                                  learning_rate=learning_rate, booster=booster, gamma=gamma,
                                  reg_alpha=reg_alpha, reg_lambda=reg_lambda, use_label_encoder=False, eval_metric='logloss')
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        return f1_score(y_train, y_pred)

    def optimize_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        best_models = {}
        
        for model_name in self.models.keys():
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self.objective(trial, model_name, X_train, y_train), n_trials=50)
            best_models[model_name] = study.best_params
        
        return best_models

    def run_optimization(self, df):
        X_buy = df.drop(columns=['y_buy', 'y_sell'])
        y_buy = df['y_buy']
        y_sell = df['y_sell']

        best_models_buy = self.optimize_model(X_buy, y_buy)
        best_models_sell = self.optimize_model(X_buy, y_sell)

        # Entrenar VotingClassifier con los mejores modelos encontrados
        voting_clf_buy = VotingClassifier(estimators=[
            ('LogisticRegression', LogisticRegression(**best_models_buy['LogisticRegression'], max_iter=30000)),
            ('SVC', SVC(**best_models_buy['SVC'])),
            ('XGB', XGBClassifier(**best_models_buy['XGB'], use_label_encoder=False, eval_metric='logloss'))
        ], voting='hard')
        
        voting_clf_sell = VotingClassifier(estimators=[
            ('LogisticRegression', LogisticRegression(**best_models_sell['LogisticRegression'], max_iter=30000)),
            ('SVC', SVC(**best_models_sell['SVC'])),
            ('XGB', XGBClassifier(**best_models_sell['XGB'], use_label_encoder=False, eval_metric='logloss'))
        ], voting='hard')

        voting_clf_buy.fit(X_buy, y_buy)
        voting_clf_sell.fit(X_buy, y_sell)

        return voting_clf_buy, voting_clf_sell

    def get_results(self):
        results = Parallel(n_jobs=-1)(delayed(self.run_optimization)(df) for df in self.dataframes)
        return results


