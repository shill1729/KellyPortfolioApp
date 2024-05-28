import yfinance as yf
import pandas as pd
from pandas.tseries.offsets import BDay
import finnhub
import streamlit as st


def get_quote(ticker):
    finnhub_client = finnhub.Client(api_key=st.secrets["FINNHUB_KEY"])
    quote = finnhub_client.quote(ticker)["c"]
    return quote


def time_to_expiration_series(expiration_dates):
    # Get the current date
    current_date = pd.Timestamp.today().normalize()

    # Convert expiration_dates to Timestamps if they are not already
    expiration_dates = pd.to_datetime(expiration_dates)

    # Calculate the number of business days between the current date and each expiration date
    business_days = expiration_dates.apply(lambda x: len(pd.date_range(start=current_date, end=x, freq=BDay())))

    # Convert business days to trading years
    trading_years = business_days / 252.0

    return trading_years


def get_option_chain(ticker, otm=False):
    # Download option chain data
    stock = yf.Ticker(ticker)
    option_dates = stock.options
    options_data = []

    for date in option_dates:
        options = stock.option_chain(date)
        calls = options.calls
        puts = options.puts
        # Combine calls and puts data
        calls['option_type'] = 'call'
        puts['option_type'] = 'put'
        options_combined = pd.concat([calls, puts])
        options_combined['expirationDate'] = date

        options_data.append(options_combined)

    options_data = pd.concat(options_data)
    options_data["mid"] = (options_data["bid"] + options_data["ask"]) / 2
    options_data["tte"] = time_to_expiration_series(options_data["expirationDate"])
    if otm:
        return options_data[options_data["inTheMoney"] == False]
    else:
        return options_data


def print_options_data(options_data):
    # Select relevant columns
    columns = ['expirationDate', 'tte', 'strike', 'lastPrice', 'mid', 'impliedVolatility', 'option_type']
    options_filtered = options_data[columns]
    # Display the data
    print(options_filtered)


if __name__ == "__main__":
    # Example usage
    ticker = 'AAPL'
    options_data = get_option_chain(ticker)
    print_options_data(options_data)

    # Example usage for a single expiration date
    expiration_date = '2024-12-31'
    time_to_exp = time_to_expiration_series(pd.Series([expiration_date]))
    print(f"Time to expiration in trading years for {expiration_date}: {time_to_exp.iloc[0]}")

    # Example usage for the expiration dates in the options data
    print(options_data["expirationDate"].iloc[0])
    # Note we have to pass a series--this throws error if use iloc[0] instead of iloc[:1]
    # but obviously, the time-to-expirations are already computed once the data is downloaded,
    # so other users won't have to worry about this when doing computations.
    time_to_exp = time_to_expiration_series(options_data["expirationDate"].iloc[:1])
    print(f"Time to expiration in trading years for first expiration date: {time_to_exp.iloc[0]}")
