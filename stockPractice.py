import yfinance as yf
import numpy as np
from datetime import datetime

# Set the start and end dates for the stock data (one month)
start = datetime(2024, 8, 1)
end = datetime(2024, 10, 1)

# Fetch the stock data for Taiwan Semiconductor (TSM) using yfinance
stock_data = yf.download('TSM', start=start, end=end)

# Calculate 20-day Exponential Moving Average (EMA)
stock_data['EMA_20'] = stock_data['Adj Close'].ewm(span=20, adjust=False).mean()

# Calculate 10-day Simple Moving Average (SMA)
stock_data['SMA_10'] = stock_data['Adj Close'].rolling(window=10).mean()

# Initialize columns for portfolio simulation
initial_capital = 10000.0  # Total initial capital
stock_data['Signal'] = 0  # Neutral signal
stock_data['Holdings'] = 0.0
stock_data['Cash'] = initial_capital
stock_data['Total'] = initial_capital
stock_data['Returns'] = np.nan

# Define buy/sell thresholds and periods
threshold = 0.01  # 1% threshold for entry/exit
lookback_period = 2  # Look for 2 consecutive days of crossover

# Create signals with a lookback period
for i in range(lookback_period, len(stock_data)):
    if stock_data['EMA_20'].iloc[i] > stock_data['SMA_10'].iloc[i] and \
       stock_data['EMA_20'].iloc[i-1] > stock_data['SMA_10'].iloc[i-1]:  # Buy signal
        stock_data['Signal'].iloc[i] = 1  # Buy signal
    elif stock_data['EMA_20'].iloc[i] < stock_data['SMA_10'].iloc[i] and \
         stock_data['EMA_20'].iloc[i-1] < stock_data['SMA_10'].iloc[i-1]:  # Sell signal
        stock_data['Signal'].iloc[i] = -1  # Sell signal

# Simulate portfolio based on signals
for i in range(1, len(stock_data)):
    prev_cash = stock_data['Cash'].iloc[i - 1]
    prev_holdings = stock_data['Holdings'].iloc[i - 1]
    adj_close = stock_data['Adj Close'].iloc[i]
    
    if stock_data['Signal'].iloc[i] == 1:  # Buy signal
        shares_to_buy = prev_cash // adj_close
        investment_value = shares_to_buy * adj_close
        stock_data.loc[stock_data.index[i], 'Holdings'] = prev_holdings + investment_value
        stock_data.loc[stock_data.index[i], 'Cash'] = prev_cash - investment_value
    elif stock_data['Signal'].iloc[i] == -1:  # Sell signal
        stock_data.loc[stock_data.index[i], 'Holdings'] = 0.0
        stock_data.loc[stock_data.index[i], 'Cash'] = prev_cash + prev_holdings
    else:  # No signal, carry forward
        stock_data.loc[stock_data.index[i], 'Holdings'] = prev_holdings
        stock_data.loc[stock_data.index[i], 'Cash'] = prev_cash
    
    # Update total portfolio value
    stock_data.loc[stock_data.index[i], 'Total'] = stock_data['Cash'].iloc[i] + stock_data['Holdings'].iloc[i]

# Calculate percentage returns on total portfolio value
stock_data['Returns'] = stock_data['Total'].pct_change()

# Display key columns
print(stock_data[['Adj Close', 'EMA_20', 'SMA_10', 'Signal', 'Holdings', 'Cash', 'Total', 'Returns']])
print("Final Portfolio Value: ${:.2f}".format(stock_data['Total'].iloc[-1]))
r