import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# List of US stocks
us_stocks = [
    'AAPL',  # Apple Inc.
    'MSFT',  # Microsoft Corporation
    'GOOGL', # Alphabet Inc. (Google)
    'AMZN',  # Amazon.com, Inc.
    'TSLA',  # Tesla, Inc.
    'META',  # Meta Platforms, Inc. (formerly Facebook)
    'NVDA',  # NVIDIA Corporation
    'BRK.B', # Berkshire Hathaway Inc. (Class B)
    'JPM',   # JPMorgan Chase & Co.
    'UNH',   # UnitedHealth Group Incorporated
    'V',     # Visa Inc.
    'MA',    # Mastercard Incorporated
    'HD',    # Home Depot, Inc.
    'DIS',   # The Walt Disney Company
    'NFLX',  # Netflix, Inc.
    'INTC',  # Intel Corporation
    'CSCO',  # Cisco Systems, Inc.
    'PFE',   # Pfizer Inc.
    'BA',    # Boeing Company
    'WMT',   # Walmart Inc.
    'PDD',   # Pinduoduo Inc.
    'FUTU'   # Futu Holdings Limited
]

today = datetime.datetime.today()

# Function to fetch historical data
def fetch_data(ticker, start_date, end_date):
    # Download historical data
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to add features for the model (lagged prices)
def add_lagged_features(data, lags=5):
    for lag in range(1, lags+1):
        data[f'lag_{lag}'] = data['Close'].shift(lag)
    data.dropna(inplace=True)
    return data


# Function to calculate simple moving averages
def calculate_sma(data, windows):
    sma = {}
    for window in windows:
        sma[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
    return sma

# Function to calculate multiple RSIs
def calculate_rsi(data, windows=[14, 28, 56]):
    rsis = {}
    for window in windows:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsis[f'RSI_{window}'] = rsi
    return rsis

def fetch_market_cap_and_shares(ticker):
    stock = yf.Ticker(ticker)
    market_cap = stock.info['marketCap']
    shares_outstanding = stock.info['sharesOutstanding']
    return market_cap, shares_outstanding

def fetch_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No data fetched. Check the ticker symbol or date range.")
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Function to print data diagnostics
def print_data_diagnostics(data):
    print("Number of Missing Values in each column:\n")
    print(data.isna().sum(), '\n\n')

    print(f"Number of Duplicated Values in the dataset: {data.duplicated().sum()}\n\n")

    print("Unique Values:\n")
    for col in data:
        print(col)
        print(data[col].unique())
        print('\n\n')

    print("Overview of Data:\n")
    print(data.describe(), '\n\n')

# Function to remove outliers from a specific column
def remove_outliers(data, column_name):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]

# Function to train a Linear Regression model and predict the next day's price
def add_lagged_features(data, lags=5):
    for lag in range(1, lags + 1):
        data[f'lag_{lag}'] = data['Close'].shift(lag)
    data.dropna(inplace=True)
    return data

# Add technical indicators (EMA, MACD, Bollinger Bands)
def add_technical_indicators(data):
    # Exponential Moving Averages (EMA)
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    
    # Moving Average Convergence Divergence (MACD)
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
    data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()
    
    data.dropna(inplace=True)
    return data

# Feature Scaling
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Function to train a Ridge Regression model and predict the next day's price
def predict_next_day_price(data):
    # Add lagged features and technical indicators
    data = add_lagged_features(data, lags=10)
    data = add_technical_indicators(data)

    # Define features and target
    features = [f'lag_{i}' for i in range(1, 11)] + ['EMA_12', 'EMA_26', 'MACD', 'BB_Middle', 'BB_Upper', 'BB_Lower']
    X = data[features]
    y = data['Close']

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    model = Ridge(alpha=1.0)
    rmse_list = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Scale features
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        # Train the model
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_list.append(rmse)

    # Print average RMSE
    avg_rmse = np.mean(rmse_list)
    print(f"Average RMSE across folds: {avg_rmse:.2f}%")

    # Predict the next day's price
    X_last_scaled = scale_features(X.iloc[-1:].values, X.iloc[-1:].values)[0]
    next_day_prediction = model.predict(X_last_scaled)[0]

    return next_day_prediction


# Function to predict next week's price movement based on SMA and RSI
def predict_price_movement(data, sma, rsis):
    # Current closing price
    current_price = data['Close'][-1]
   
    # Latest SMA values
    sma_50 = sma['SMA_50'][-1]
    sma_200 = sma['SMA_200'][-1]
   
    # Latest RSI values
    rsi_14 = rsis['RSI_14'][-1]
   
    # Price movement prediction
    prediction = ""

    # SMA-based prediction
    if current_price > sma_50 and current_price > sma_200:
        prediction += "The stock is in a bullish trend based on the SMAs. "
    elif current_price < sma_50 and current_price < sma_200:
        prediction += "The stock is in a bearish trend based on the SMAs. "
    else:
        prediction += "The stock shows no clear trend based on the SMAs. "

    # RSI-based prediction
    if rsi_14 > 70:
        prediction += "The RSI indicates the stock is overbought, suggesting a possible price correction."
    elif rsi_14 < 30:
        prediction += "The RSI indicates the stock is oversold, suggesting a potential price recovery."
    else:
        prediction += "The RSI indicates the stock is neither overbought nor oversold, suggesting no extreme conditions."
   
    return prediction

def compare_market_cap(data, shares_outstanding, current_market_cap):
    # Calculate historical market capitalization (1 year ago, if data is available)
    one_year_ago_price = data['Close'][-252]  # Assuming 252 trading days in a year
    one_year_ago_market_cap = one_year_ago_price * shares_outstanding
   
    # Compare current vs one year ago
    if current_market_cap > one_year_ago_market_cap:
        return f"Market cap increased compared to 1 year ago. ({one_year_ago_market_cap / 1e9:.2f}B -> {current_market_cap / 1e9:.2f}B)"
    else:
        return f"Market cap decreased compared to 1 year ago. ({one_year_ago_market_cap / 1e9:.2f}B -> {current_market_cap / 1e9:.2f}B)"


# Function to plot data and show predictions
def plot_data_with_prediction(data, sma, rsis, ticker, prediction, next_day_price, market_cap_message):

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(14, 14), sharex=True)

    # Plot closing price and simple moving averages on the first subplot
    ax1.plot(data.index, data['Close'], label='Close Price', color='red')
    for sma_name, sma_series in sma.items():
        ax1.plot(sma_series, label=sma_name)

    ax1.set_title(f'{ticker} Price and Simple Moving Averages')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Add the prediction text and next day price prediction on the plot
    ax1.text(0.12, 0.95, f"$\mathbf{{Next\ Day\ Predicted\ Price:}} {next_day_price:.2f}$\n \nPrediction: {prediction}  \n{market_cap_message}\nToday date:{today:%B %d, %Y}", 
             transform=ax1.transAxes, fontsize=8, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot RSI_14 on the second subplot
    ax2.plot(rsis['RSI_14'], label='RSI_14', color='orange', linestyle='--')
    ax2.axhline(30, linestyle='--', color='red', alpha=0.5)
    ax2.axhline(70, linestyle='--', color='green', alpha=0.5)
    ax2.set_ylabel('RSI_14')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    # Plot RSI_28 and RSI_56 on the third subplot
    ax3.plot(rsis['RSI_28'], label='RSI_28', color='purple', linestyle='--')
    ax3.plot(rsis['RSI_56'], label='RSI_56', color='green', linestyle='--')
    ax3.axhline(30, linestyle='--', color='red', alpha=0.5)
    ax3.axhline(70, linestyle='--', color='green', alpha=0.5)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('RSI')
    ax3.legend(loc='upper left')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


# Main function to execute the program
def main():
    print("Available stocks:")
    for stock in us_stocks:
        print(stock)
    ticker = input("Enter the stock ticker you want to see: ")
    start_date = '2014-09-01'
    end_date = '2024-10-3'
    windows = [20, 50, 200]  # Simple moving average windows

    # Fetch data
    data = fetch_data(ticker, start_date, end_date)
    if data.empty:
        return

    # Print data diagnostics
    print_data_diagnostics(data)

    # Remove outliers
    columns_to_clean = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in columns_to_clean:
        data = remove_outliers(data, col)

    print(data.head())
    print(f"Cleaned Data (After Removing Outliers): {data.shape}\n\n")

    # Check if data is not empty after cleaning
    if not data.empty:
        # Calculate the correlation matrix
        correlation_matrix = data[['Open', 'Close', 'Volume']].corr()
        
        # Print the correlation matrix
        print("Correlation Matrix:\n", correlation_matrix)

        # Visualize the correlation matrix
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Stock Variables')
        plt.xlabel('Variables')
        plt.ylabel('Variables')
        plt.show()

    # Calculate simple moving averages
    sma = calculate_sma(data, windows)

    # Calculate RSIs
    rsis = calculate_rsi(data)

    # Fetch market capitalization and shares outstanding
    current_market_cap, shares_outstanding = fetch_market_cap_and_shares(ticker)

    # Predict next week's price movement
    prediction = predict_price_movement(data, sma, rsis)
    print("\nPrice movement prediction for next week:")
    print(prediction)

    # Predict the next day's price using the Linear Regression model
    next_day_price = predict_next_day_price(data)
    print(f"\n\033[1m Next day's predicted price: {next_day_price:.2f}\033[0m")

    # Market capitalization comparison
    market_cap_message = compare_market_cap(data, shares_outstanding, current_market_cap)
    print(f"Market capitalization comparison: {market_cap_message}")

    # Plot data, simple moving averages, and prediction
    plot_data_with_prediction(data, sma, rsis, ticker, prediction, next_day_price, market_cap_message)

if __name__ == "__main__":
    main()