# Importing libraries
from scipy.stats import ks_2samp
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data_train = pd.read_csv("./data/aapl_5m_train.csv").dropna()
data_test = pd.read_csv("./data/aapl_5m_test.csv").dropna()

# Normalize data
train_mean = data_train.loc[:, ["Open", "High", "Low", "Close"]].mean()
train_std = data_train.loc[:, ["Open", "High", "Low", "Close"]].std()
norm_data_train = (data_train.loc[:, ["Open", "High", "Low", "Close"]] - train_mean) / train_std
norm_data_test = (data_test.loc[:, ["Open", "High", "Low", "Close"]] - train_mean) / train_std

# Generate features
lags = 5
X_train = pd.DataFrame()
X_test = pd.DataFrame()

for lag in range(lags):
    X_train[f"Open_{lag}"] = norm_data_train.Open.shift(lag)
    X_train[f"High_{lag}"] = norm_data_train.High.shift(lag)
    X_train[f"Low_{lag}"] = norm_data_train.Low.shift(lag)
    X_train[f"Close_{lag}"] = norm_data_train.Close.shift(lag)
    
    X_test[f"Open_{lag}"] = norm_data_test.Open.shift(lag)
    X_test[f"High_{lag}"] = norm_data_test.High.shift(lag)
    X_test[f"Low_{lag}"] = norm_data_test.Low.shift(lag)
    X_test[f"Close_{lag}"] = norm_data_test.Close.shift(lag)

Y_train = (X_train.Close_0 * (1 + 0.01) < X_train.Close_0.shift(-1)).astype(float)
Y_test = (X_test.Close_0 * (1 + 0.01) < X_test.Close_0.shift(-1)).astype(float)

# Removing nans and last value
X_train = X_train.iloc[5:-1, :].values
X_test = X_test.iloc[5:-1, :].values

Y_train = Y_train.iloc[5:-1].values.reshape(-1, 1)
Y_test = Y_test.iloc[5:-1].values.reshape(-1, 1)

# Reshaping tensors
features = X_train.shape[1]
X_train = X_train.reshape(-1, features, 1)
X_test = X_test.reshape(-1, features, 1)

# Define transformer model
def create_transformer():
    input_shape = X_train.shape[1:]

    # Hyperparameters
    head_size = 256
    num_heads = 4
    num_transformer_blocks = 4
    dnn_dim = 4
    units = 128

    # Define input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    # Stacking transformer blocks
    for _ in range(num_transformer_blocks):
        x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=0.2)(x, x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Add()([x, inputs])

        # DNN layers
        x = tf.keras.layers.Conv1D(filters=4, kernel_size=1, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Add()([x, inputs])

    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # MLP layers
    x = tf.keras.layers.Dense(units=128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(units=128, activation="relu")(x)

    # Output layer
    outputs = tf.keras.layers.Dense(units=2, activation="softmax")(x)

    # Model
    model = tf.keras.Model(inputs, outputs)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model

# Train model
def train_model(model, X_train, Y_train):
    model.fit(X_train, Y_train, epochs=10, batch_size=64)

# Backtest model
def backtest(model, X_test, trading_data):
    capital = 1_000_000
    n_shares = 40
    stop_loss = 0.10
    take_profit = 0.10
    COM = 0.025
    active_positions = []
    portfolio_value = [capital]

    for i, row in trading_data.iterrows():
        yhat = model.predict(X_test[i:i+1])[0, 1] > 0.5
        if yhat:
            if capital > row.Close * (1 + COM) * n_shares:
                capital -= row.Close * (1 + COM) * n_shares
                active_positions.append({
                    "type": "LONG",
                    "bought_at": row.Close,
                    "n_shares": n_shares,
                    "stop_loss": row.Close * (1 - stop_loss),
                    "take_profit": row.Close * (1 + take_profit)
                })
        positions_value = len(active_positions) * n_shares * row.Close
        portfolio_value.append(capital + positions_value)

    return portfolio_value

# Load and process train and test data
model = create_transformer()
train_model(model, X_train, Y_train)

# Perform KS test and retrain if necessary
test_prices = []

for i in range(len(data_test)):
    yhat = model.predict(X_test[i:i+1])[0, 1] > 0.5
    test_prices.append(data_test.iloc[i].Close)
    if len(test_prices) >= len(data_test):
        break
    ks_stat, p_value = ks_2samp(data_train.Close, test_prices)
    if p_value < 0.05:
        model = create_transformer()
        train_model(model, X_train, Y_train)

# Backtest and plot results
portfolio_value = backtest(model, X_test, data_test)
plt.plot(portfolio_value)
plt.show()
