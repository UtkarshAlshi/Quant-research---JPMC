import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.interpolate import interp1d

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'])
    data.set_index('Date', inplace=True)
    return data.sort_index()

# Visualize existing price data
def plot_data(data, title, extrapolated_data=None):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Price'], marker='o', label='Historical Prices')
    if extrapolated_data is not None:
        plt.plot(extrapolated_data.index, extrapolated_data['Price'], 
                 marker='x', linestyle='--', label='Extrapolated Prices')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Interpolation function
def estimate_price(data, date_str):
    date = pd.to_datetime(date_str)
    if date < data.index.min() or date > data.index.max():
        return "Date out of range for interpolation."
    f = interp1d(data.index.astype(np.int64), data['Price'], kind='linear', fill_value='extrapolate')
    return f(date.astype(np.int64))

# Extrapolation function
def extrapolate_future_prices(data, months_ahead=12):
    future_dates = pd.date_range(data.index.max() + timedelta(days=1), periods=months_ahead, freq='M')
    x = np.arange(len(data))
    poly_fit = np.polyfit(x, data['Price'].values, deg=3)
    future_x = np.arange(len(data), len(data) + months_ahead)
    future_prices = np.polyval(poly_fit, future_x)
    return pd.DataFrame({'Price': future_prices}, index=future_dates)

def main():
    # Load data
    data = load_data('natural_gas_prices.csv')

    # Plot historical data
    plot_data(data, 'Natural Gas Prices (Oct 2020 - Sep 2024)')

    # Estimate price for a specific date
    estimated_price = estimate_price(data, '2023-05-15')
    print(f"Estimated price on 2023-05-15: {estimated_price:.2f} USD")

    # Extrapolate future prices and plot
    future_data = extrapolate_future_prices(data)
    plot_data(data, 'Natural Gas Prices (Historical and Extrapolated)', future_data)

if __name__ == "__main__":
    main()