import pandas as pd
import ta

# Cargar los datos históricos desde el CSV
aapl_1m = pd.read_csv('../aapl_project_1m_train.csv')

# Asegurarse de que la columna 'Close' esté presente
if 'Close' not in aapl_1m.columns:
    raise ValueError("La columna 'Close' no está presente en el DataFrame")

# Calcular el RSI (Relative Strength Index)
aapl_1m['RSI'] = ta.momentum.RSIIndicator(aapl_1m['Close'], window=14).rsi()

# Calcular el ATR (Average True Range)
aapl_1m['ATR'] = ta.volatility.AverageTrueRange(
    high=aapl_1m['High'],
    low=aapl_1m['Low'],
    close=aapl_1m['Close'],
    window=14
).average_true_range()

# Calcular el Williams %R
aapl_1m['Williams_%R'] = ta.momentum.WilliamsRIndicator(
    high=aapl_1m['High'],
    low=aapl_1m['Low'],
    close=aapl_1m['Close'],
    lbp=14
).williams_r()

# Generar señales de compra y venta
def generate_signals(df):
    df['Buy_Signal'] = ((df['RSI'] < 30) & (df['Williams_%R'] < -80)).astype(int)
    df['Sell_Signal'] = ((df['RSI'] > 70) & (df['Williams_%R'] > -20)).astype(int)
    return df

# Aplicar la función para generar señales
aapl_1m = generate_signals(aapl_1m)

# Guardar el DataFrame con las señales en un nuevo CSV
aapl_1m.to_csv('aapl_signals.csv', index=False)

# Mostrar las primeras filas del DataFrame con las señales
print(aapl_1m.head())
