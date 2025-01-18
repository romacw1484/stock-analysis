import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go


# 1. Download Stock Data
def download_stock_data(ticker, start_date, end_date):
    """
    Downloads historical stock data from Yahoo Finance.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data



def visualize_stock_data(data, symbol, start_date, end_date):
    """
    Visualizes the stock's closing price with a dark theme.
    Handles MultiIndex columns for ticker-specific data.
    """
    # Check if the data is empty
    if data.empty:
        print(f"No data found for {symbol}. Cannot plot the data.")
        return

    # Handle MultiIndex columns: Select the specific ticker's data if necessary
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.xs(symbol, axis=1, level='Ticker')
        except KeyError:
            print(f"No data found for ticker {symbol}.")
            return

    # Debugging: Print cleaned data
    print("Cleaned Data Columns:", data.columns)
    print("Cleaned Sample Data:\n", data.head())

    # Check if 'Close' column exists after processing
    if 'Close' not in data.columns:
        print(f"'Close' column not found for {symbol}. Cannot plot the data.")
        return

    # Reset the index and ensure 'Date' column exists
    data = data.reset_index()

    # Ensure 'Date' is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Remove rows with NaN values in 'Close' to avoid plotting issues
    data = data.dropna(subset=['Close'])

    # Create the plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['Close'],
            line=dict(color='lime', width=2),
            name=f'{symbol} Close Price'
        )
    )

    # Add annotations for start and end points
    fig.add_annotation(
        x=data['Date'].iloc[0],
        y=data['Close'].iloc[0],
        text='Start',
        showarrow=True,
        arrowhead=2,
        arrowcolor='red',
        font=dict(color='red')
    )
    fig.add_annotation(
        x=data['Date'].iloc[-1],
        y=data['Close'].iloc[-1],
        text='End',
        showarrow=True,
        arrowhead=2,
        arrowcolor='green',
        font=dict(color='green')
    )

    # Update layout for dark theme
    fig.update_layout(
        title=f'{symbol} Stock Price from {start_date} to {end_date}',
        title_font=dict(size=22, color='white', family='Optima, sans-serif'),
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        plot_bgcolor='#1f1f1f',
        paper_bgcolor='#1f1f1f',
        font=dict(color='white'),
        xaxis=dict(
            showgrid=True,
            gridcolor='gray',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='gray',
            zeroline=False
        ),
        margin=dict(l=60, r=40, t=80, b=50),
        title_x=0.5  # Center the title
    )
    fig.show()

def compare_stock_prices(symbol1, symbol2, start_date, end_date):
    """
    Compares the stock prices of two stocks by plotting their closing prices.
    """
    # Download stock data for both stocks
    data1 = download_stock_data(symbol1, start_date, end_date)
    data2 = download_stock_data(symbol2, start_date, end_date)

    # Check if data is empty for either stock
    if data1.empty:
        print(f"No data found for {symbol1}. Cannot compare.")
        return
    if data2.empty:
        print(f"No data found for {symbol2}. Cannot compare.")
        return

    # Handle MultiIndex columns if present
    if isinstance(data1.columns, pd.MultiIndex):
        data1 = data1.xs(symbol1, axis=1, level='Ticker')
    if isinstance(data2.columns, pd.MultiIndex):
        data2 = data2.xs(symbol2, axis=1, level='Ticker')

    # Ensure 'Close' column exists for both datasets
    if 'Close' not in data1.columns or 'Close' not in data2.columns:
        print(f"One of the datasets is missing the 'Close' column. Cannot compare.")
        return

    # Reset index and clean data
    data1 = data1.reset_index()
    data2 = data2.reset_index()
    data1['Date'] = pd.to_datetime(data1['Date'])
    data2['Date'] = pd.to_datetime(data2['Date'])
    data1 = data1.dropna(subset=['Close'])
    data2 = data2.dropna(subset=['Close'])

    # Create the plot
    fig = go.Figure()

    # Add stock 1 data
    fig.add_trace(
        go.Scatter(
            x=data1['Date'],
            y=data1['Close'],
            line=dict(color='blue', width=2),
            name=f'{symbol1} Close Price'
        )
    )

    # Add stock 2 data
    fig.add_trace(
        go.Scatter(
            x=data2['Date'],
            y=data2['Close'],
            line=dict(color='orange', width=2),
            name=f'{symbol2} Close Price'
        )
    )

    # Update layout for dark theme
    fig.update_layout(
        title=f'{symbol1} vs {symbol2} Stock Prices from {start_date} to {end_date}',
        title_font=dict(size=22, color='white', family='Arial Black'),
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        plot_bgcolor='#1f1f1f',
        paper_bgcolor='#1f1f1f',
        font=dict(color='white'),
        xaxis=dict(
            showgrid=True,
            gridcolor='gray',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='gray',
            zeroline=False
        ),
        margin=dict(l=60, r=40, t=80, b=50),
        title_x=0.5  # Center the title
    )

    fig.show()

def moving_average_crossover_strategy(data, symbol):
    """
    Implements the Moving Average Crossover Strategy.
    Uses 20-day and 50-day moving averages to generate buy/sell signals.
    Simulates trading with an initial investment of $5000.
    """
    # Ensure 'Close' exists and handle MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(symbol, axis=1, level='Ticker')

    if 'Close' not in data.columns:
        print(f"No 'Close' column found for {symbol}. Cannot perform strategy.")
        return

    # Calculate the 20-day and 50-day moving averages
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()

    # Generate signals: 1 for Buy, 0 for Hold, -1 for Sell
    data['Signal'] = 0
    data.loc[data['MA_20'] > data['MA_50'], 'Signal'] = 1  # Buy signal
    data.loc[data['MA_20'] <= data['MA_50'], 'Signal'] = -1  # Sell signal

    # Simulate trading with an initial investment
    initial_investment = 5000
    cash = initial_investment
    stock_holding = 0
    trade_size_pct = 0.1  # 10% of portfolio per trade

    portfolio_values = []

    for i in range(len(data)):
        price = data['Close'].iloc[i]
        signal = data['Signal'].iloc[i]

        if signal == 1:  # Buy signal
            trade_size = cash * trade_size_pct
            shares_to_buy = trade_size / price
            stock_holding += shares_to_buy
            cash -= trade_size

        elif signal == -1 and stock_holding > 0:  # Sell signal
            cash += stock_holding * price
            stock_holding = 0

        # Track portfolio value
        portfolio_value = cash + (stock_holding * price)
        portfolio_values.append(portfolio_value)

    # Add portfolio value to data
    data['Portfolio_Value'] = portfolio_values

    # Create the plot
    fig = go.Figure()

    # Plot green and red segments for Close Price
    for i in range(1, len(data)):
        color = 'green' if data['Signal'].iloc[i] == 1 else 'red'
        fig.add_trace(
            go.Scatter(
                x=data.index[i-1:i+1],
                y=data['Close'].iloc[i-1:i+1],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False
            )
        )

    # Plot the 20-day and 50-day moving averages
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MA_20'],
            mode='lines',
            name='20-Day MA',
            line=dict(color='orange', width=2, dash='dot')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MA_50'],
            mode='lines',
            name='50-Day MA',
            line=dict(color='green', width=2, dash='dot')
        )
    )

    # Second Graph: Portfolio Value
    fig2 = go.Figure()

    # Plot portfolio value
    fig2.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Portfolio_Value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        )
    )

    # Update layout for the second graph
    fig2.update_layout(
        title=f'{symbol} - Portfolio Value Over Time',
        title_font=dict(size=22, color='white', family='Arial Black'),
        xaxis_title='Date',
        yaxis_title='Portfolio Value',
        plot_bgcolor='#1f1f1f',
        paper_bgcolor='#1f1f1f',
        font=dict(color='white'),
        xaxis=dict(
            showgrid=True,
            gridcolor='gray',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='gray',
            zeroline=False
        ),
        margin=dict(l=60, r=40, t=80, b=50),
        title_x=0.5  # Center the title
    )

    fig2.show()


    # Update layout
    fig.update_layout(
        title=f'{symbol} - Moving Average Crossover Strategy (Portfolio Simulation)',
        title_font=dict(size=22, color='white', family='Arial Black'),
        xaxis_title='Date',
        yaxis_title='Price / Portfolio Value',
        plot_bgcolor='#1f1f1f',
        paper_bgcolor='#1f1f1f',
        font=dict(color='white'),
        xaxis=dict(
            showgrid=True,
            gridcolor='gray',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='gray',
            zeroline=False
        ),
        margin=dict(l=60, r=40, t=80, b=50),
        title_x=0.5  # Center the title
    )

    fig.show()

   # Calculate daily and strategy returns
    data['Daily_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Signal'].shift(1) * data['Daily_Return']

    # Calculate win ratio
    valid_trades = data['Strategy_Return'].notna() & (data['Signal'].shift(1) != 0)  # Exclude invalid or initial zero signals
    win_trades = (data.loc[valid_trades, 'Strategy_Return'] > 0).sum()
    total_trades = valid_trades.sum()
    win_ratio = (win_trades / total_trades) * 100 if total_trades > 0 else 0

    # Calculate Buy-and-Hold Return
    initial_price = data['Close'].iloc[0]
    final_price = data['Close'].iloc[-1]
    buy_and_hold_return = (final_price / initial_price) * initial_investment

    # Final portfolio value
    final_portfolio_value = portfolio_values[-1]

    # Print results
    print(f"Final Portfolio Value: ${final_portfolio_value:.2f} (Initial Investment: ${initial_investment:.2f})")
    print(f"Win Ratio for Algo: {win_ratio:.2f}%")
    print(f"Buy-and-Hold Return: ${buy_and_hold_return:.2f}")
    print(
        f"Performance Comparison: {'ALGO OUTPERFORMED' if final_portfolio_value > buy_and_hold_return else 'BUY-AND-HOLD OUTPERFORMED'}"
    )



# Updated Main Program
def main():
    print(f"Hello, welcome to Willy's WallStreet !!! ")

    print(f'''\nWhat would you like to do: 
    A - Show me the price chart of a stock 
    B - Show me the price comparison two stocks
    C - Show me trading strategies given a stock 
    D - Show me the cash flow statement of my chosen stock
    E - Show me the analyst recommendations for the stock of the last 6 months
    F - Show me the price prediction for the next 30 days
    Q - Quit the program
    ''')
    # Get user choice
    choice = input('Please enter your choice (e.g. for the first option type "A"): ').strip().upper()

    # Quit option
    if choice == "Q":
        print("Thank you for using Willy's WallStreet! Goodbye.")
        return

    
    # Define a date range for stock data
    end_date = date.today()
    start_date = end_date - relativedelta(years=2)


    # Execute the corresponding function based on the user's choice
    if choice == "A":

         # Get the stock symbol for the "A" option
        symbol = input("Please enter the stock symbol: ").upper().strip()

        # Download stock data
        stock_data = download_stock_data(symbol, start_date, end_date)

        if stock_data.empty:
            print(f"No data found for {symbol} from {start_date} to {end_date}. Please try again.")
            return

        # Visualize stock data
        visualize_stock_data(stock_data, symbol, start_date, end_date)


   
    elif choice == "B":
        symbol1 = input("Please enter the first stock symbol: ").upper().strip()
        symbol2 = input("Please enter the second stock symbol: ").upper().strip()
        compare_stock_prices(symbol1, symbol2, start_date, end_date)

    elif choice == "C":
        symbol = input("Please enter the stock symbol: ").upper().strip()
        stock_data = download_stock_data(symbol, start_date, end_date)

        if stock_data.empty:
            print(f"No data found for {symbol} from {start_date} to {end_date}. Please try again.")
            return

        moving_average_crossover_strategy(stock_data, symbol)


        
    '''
        
    elif choice == "C":
        show_revenue_and_earnings(symbol)  # Placeholder function for revenue and earnings
    elif choice == "D":
        show_cash_flow_statement(symbol)  # Placeholder function for cash flow statement
    elif choice == "E":
        show_analyst_recommendations(symbol)  # Placeholder function for analyst recommendations
    elif choice == "F":
        predict_stock_prices(stock_data, symbol)  # Placeholder function for price prediction
    else:
        print("Invalid choice. Please try again.")
        return
        '''

    print("Thank you for using Willy's WallStreet!")


if __name__ == "__main__":
    main()
