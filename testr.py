import numpy as np
import pandas as pd
import yfinance as yf
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Data Retrieval and Processing
def get_stock_data(ticker):
    data = yf.download(ticker, start='2020-01-01', end='2023-10-13')
    data.reset_index(inplace=True)
    data['Close'] = data['Adj Close']
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['Signal'] = 0
    data['Signal'][20:] = np.where(data['SMA20'][20:] > data['SMA50'][20:], 1, -1)
    data['Position'] = data['Signal'].diff()
    return data

def simulate_trading(data, initial_investment=5000):
    cash = initial_investment
    stock = 0
    portfolio_values = []

    for i in range(len(data)):
        if data['Position'].iloc[i] == 2:  # Buy signal
            stock = cash / data['Close'].iloc[i]
            cash = 0
        elif data['Position'].iloc[i] == -2:  # Sell signal
            cash = stock * data['Close'].iloc[i]
            stock = 0
        portfolio_value = cash + stock * data['Close'].iloc[i]
        portfolio_values.append(portfolio_value)

    data['Portfolio Value'] = portfolio_values
    return data

# Initialize the app
app = dash.Dash(__name__)
server = app.server  # For deployment

# Get the data
ticker = 'F'
data = get_stock_data(ticker)
data = simulate_trading(data)

# App Layout
app.layout = html.Div([
    html.H1('TSM Stock Analysis Dashboard'),
    dcc.Graph(id='price-chart'),
    dcc.Graph(id='portfolio-chart'),
    html.Div([
        html.Label('Initial Investment ($):'),
        dcc.Input(
            id='initial-investment',
            type='number',
            value=5000,
            min=1000,
            step=500
        )
    ], style={'width': '48%', 'display': 'inline-block'}),
])

# Callbacks
@app.callback(
    [Output('price-chart', 'figure'),
     Output('portfolio-chart', 'figure')],
    [Input('initial-investment', 'value')]
)
def update_graph(initial_investment):
    # Re-simulate trading with new initial investment
    updated_data = simulate_trading(data.copy(), initial_investment=initial_investment)

    # Price Chart with Buy/Sell Signals
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(
        x=updated_data['Date'],
        y=updated_data['Close'],
        mode='lines',
        name='Close Price'
    ))
    price_fig.add_trace(go.Scatter(
        x=updated_data['Date'],
        y=updated_data['SMA20'],
        mode='lines',
        name='SMA20'
    ))
    price_fig.add_trace(go.Scatter(
        x=updated_data['Date'],
        y=updated_data['SMA50'],
        mode='lines',
        name='SMA50'
    ))
    # Buy Signals
    buy_signals = updated_data[updated_data['Position'] == 2]
    price_fig.add_trace(go.Scatter(
        x=buy_signals['Date'],
        y=buy_signals['Close'],
        mode='markers',
        marker_symbol='triangle-up',
        marker_color='green',
        marker_size=10,
        name='Buy Signal'
    ))
    # Sell Signals
    sell_signals = updated_data[updated_data['Position'] == -2]
    price_fig.add_trace(go.Scatter(
        x=sell_signals['Date'],
        y=sell_signals['Close'],
        mode='markers',
        marker_symbol='triangle-down',
        marker_color='red',
        marker_size=10,
        name='Sell Signal'
    ))
    price_fig.update_layout(
        title='TSM Stock Price with Buy/Sell Signals',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified'
    )

    # Portfolio Value Chart
    portfolio_fig = go.Figure()
    portfolio_fig.add_trace(go.Scatter(
        x=updated_data['Date'],
        y=updated_data['Portfolio Value'],
        mode='lines',
        name='Portfolio Value'
    ))
    portfolio_fig.update_layout(
        title='Portfolio Value Over Time',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified'
    )

    return price_fig, portfolio_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
