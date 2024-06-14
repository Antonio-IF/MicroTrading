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
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid'])
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            model = SVC(C=C, kernel=kernel, gamma=gamma, max_iter=30000)
        elif model_name == 'XGB':
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 2, 16)
            max_leaves = trial.suggest_int('max_leaves', 0, 256)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
            booster = trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart'])
            gamma = trial.suggest_float('gamma', 0, 5)
            reg_alpha = trial.suggest_float('reg_alpha', 0, 5)
            reg_lambda = trial.suggest_float('reg_lambda', 0, 5)
            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_leaves=max_leaves,
                learning_rate=learning_rate,
                booster=booster,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        return f1_score(y_train, y_pred)

    def optimize_model(self, model_name, X_train, y_train):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, model_name, X_train, y_train), n_trials=20, n_jobs=-1)
        return study.best_params

    def train_and_evaluate(self):
        priority_datasets = ['BTC_5M_TRAIN', 'AAPL_5M_TRAIN']
        for dataset in priority_datasets + [name for name in self.dataframes if name not in priority_datasets]:
            df = self.dataframes[dataset]
            df_clean = self.preprocess_data(df)
            X = df_clean.drop(columns=["Y", "Close"])
            y_buy = (df_clean["Y"] > df_clean["Close"]).astype(int)
            y_sell = (df_clean["Y"] < df_clean["Close"]).astype(int)

            # Split for buy signals
            X_train_buy, X_val_buy, y_train_buy, y_val_buy = train_test_split(X, y_buy, test_size=0.2, random_state=42)

            self.results[dataset] = {'buy': {}, 'sell': {}}
            best_params_combined_buy = {}
            for model_name, model_cls in self.models.items():
                best_params = self.optimize_model(model_name, X_train_buy, y_train_buy)
                best_params_combined_buy[model_name] = best_params
                model = model_cls(**best_params)
                model.fit(X_train_buy, y_train_buy)
                y_pred = model.predict(X_val_buy)
                accuracy = accuracy_score(y_val_buy, y_pred)
                f1 = f1_score(y_val_buy, y_pred)
                cm = confusion_matrix(y_val_buy, y_pred)
                fpr = cm[0][1] / (cm[0][1] + cm[0][0])
                self.results[dataset]['buy'][model_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'fpr': fpr,
                    'best_params': best_params
                }

            # Create the Stocking model for buy signals
            estimators_buy = [
                ('LogisticRegression', LogisticRegression(**best_params_combined_buy['LogisticRegression'], max_iter=30000)),
                ('SVC', SVC(**best_params_combined_buy['SVC'], max_iter=30000)),
                ('XGB', XGBClassifier(**best_params_combined_buy['XGB'], use_label_encoder=False, eval_metric='mlogloss'))
            ]
            stocking_model_buy = VotingClassifier(estimators=estimators_buy, voting='hard')
            stocking_model_buy.fit(X_train_buy, y_train_buy)
            y_pred_stocking_buy = stocking_model_buy.predict(X_val_buy)
            accuracy_stocking = accuracy_score(y_val_buy, y_pred_stocking_buy)
            f1_stocking = f1_score(y_val_buy, y_pred_stocking_buy)
            cm_stocking = confusion_matrix(y_val_buy, y_pred_stocking_buy)
            fpr_stocking = cm_stocking[0][1] / (cm_stocking[0][1] + cm_stocking[0][0])

            self.results[dataset]['buy']['Stocking'] = {
                'accuracy': accuracy_stocking,
                'f1_score': f1_stocking,
                'fpr': fpr_stocking
            }

            # Split for sell signals
            X_train_sell, X_val_sell, y_train_sell, y_val_sell = train_test_split(X, y_sell, test_size=0.2, random_state=42)

            best_params_combined_sell = {}
            for model_name, model_cls in self.models.items():
                best_params = self.optimize_model(model_name, X_train_sell, y_train_sell)
                best_params_combined_sell[model_name] = best_params
                model = model_cls(**best_params)
                model.fit(X_train_sell, y_train_sell)
                y_pred = model.predict(X_val_sell)
                accuracy = accuracy_score(y_val_sell, y_pred)
                f1 = f1_score(y_val_sell, y_pred)
                cm = confusion_matrix(y_val_sell, y_pred)
                fpr = cm[0][1] / (cm[0][1] + cm[0][0])
                self.results[dataset]['sell'][model_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'fpr': fpr,
                    'best_params': best_params
                }

            # Create the Stocking model for sell signals
            estimators_sell = [
                ('LogisticRegression', LogisticRegression(**best_params_combined_sell['LogisticRegression'], max_iter=30000)),
                ('SVC', SVC(**best_params_combined_sell['SVC'], max_iter=30000)),
                ('XGB', XGBClassifier(**best_params_combined_sell['XGB'], use_label_encoder=False, eval_metric='mlogloss'))
            ]
            stocking_model_sell = VotingClassifier(estimators=estimators_sell, voting='hard')
            stocking_model_sell.fit(X_train_sell, y_train_sell)
            y_pred_stocking_sell = stocking_model_sell.predict(X_val_sell)
            accuracy_stocking = accuracy_score(y_val_sell, y_pred_stocking_sell)
            f1_stocking = f1_score(y_val_sell, y_pred_stocking_sell)
            cm_stocking = confusion_matrix(y_val_sell, y_pred_stocking_sell)
            fpr_stocking = cm_stocking[0][1] / (cm_stocking[0][1] + cm_stocking[0][0])

            self.results[dataset]['sell']['Stocking'] = {
                'accuracy': accuracy_stocking,
                'f1_score': f1_stocking,
                'fpr': fpr_stocking
            }

    def get_results(self):
        return self.results


