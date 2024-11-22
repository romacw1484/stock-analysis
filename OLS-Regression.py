import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from datetime import datetime

# 1. Fetch stock data
start = datetime(2024, 8, 1)
end = datetime(2024, 9, 1)
stock_data = yf.download('TSM', start=start, end=end)

# 2. Calculate moving averages as independent variables
stock_data['EMA_20'] = stock_data['Adj Close'].ewm(span=20, adjust=False).mean()
stock_data['SMA_10'] = stock_data['Adj Close'].rolling(window=10).mean()

# 3. Prepare the data for regression
y = stock_data['Adj Close']  # Dependent variable
X = stock_data[['EMA_20', 'SMA_10']]  # Independent variables
X = sm.add_constant(X)  # Add constant for intercept

# 4. Handle missing data
X = X.dropna()  # Drop rows with NaN in independent variables
y = y.dropna()  # Drop rows with NaN in dependent variable

# Ensure X and y are aligned after dropping rows
X, y = X.align(y, join='inner', axis=0)

# 5. Fit the OLS regression model
model = sm.OLS(y, X).fit()

# 6. Display the regression results
print(model.summary())

# 7. Predict stock prices using the regression model
stock_data['Predicted'] = model.predict(sm.add_constant(stock_data[['EMA_20', 'SMA_10']]))
print(stock_data[['Adj Close', 'Predicted']])
