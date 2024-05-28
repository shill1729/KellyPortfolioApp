import numpy as np
import streamlit as st
import datetime as dt
import pandas as pd
import alphavantage.av as av
import finnhub
import time
from scipy.stats import norm
from allocator import compute_allocations
from sdes import MultiGbm
from optimal_mispriced_option import optimal_option_strategy
from constants import rf_rate


def update_with_quotes(prices):
    today = dt.date.today()
    symbols = prices.columns
    quotes = []
    finnhub_client = finnhub.Client(api_key=st.secrets["FINNHUB_KEY"])
    for symbol in symbols:
        quote = finnhub_client.quote(symbol)["c"]
        quotes.append(quote)
        time.sleep(1 / len(symbols))
    new_row = pd.DataFrame([quotes], columns=symbols, index=[today])
    st.session_state.quotes = new_row

    # replace the row for today if it exists
    if today in prices.index:
        prices.loc[today] = quotes
    else:
        prices = pd.concat([prices, new_row])

    return prices


@st.cache_data(show_spinner=False)
def download_data(symbols):
    api = av.av()
    av_key = st.secrets["AV_API_KEY"]
    api.log_in(av_key)
    period = "daily"
    interval = None
    adjusted = True
    what = "adjusted_close"
    asset_type = "stocks"
    timescale = av.timescale(period, interval, asset_type)
    data = api.get_assets(symbols, period, interval, adjusted, what)
    now = dt.datetime.now().astimezone(dt.timezone(dt.timedelta(hours=-4)))
    start_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end_time = now.replace(hour=18, minute=0, second=0, microsecond=0)
    is_market_hours = now.weekday() < 5 and start_time <= now <= end_time
    if is_market_hours:
        data = update_with_quotes(data)
    else:
        st.write("Previous Close Prices:")
        st.write(data.iloc[-1, :])
    return data, timescale


if __name__ == "__main__":
    gbm = MultiGbm()
    st.title("Optimal Log Growth Allocations")
    default_ticker_list = "TSLA, AMC, GME, NKLA, GOTU, NVDA, AAPL, DIS, ROKU, AMZN"
    ticker_list = st.text_input("Enter a comma-separated list of tickers", default_ticker_list)
    symbols = [s.strip() for s in ticker_list.split(",")]

    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None

    if 'timescale' not in st.session_state:
        st.session_state.timescale = None

    if 'quotes' not in st.session_state:
        st.session_state.quotes = pd.DataFrame(columns=symbols)

    ema_filter = st.slider("Select the EMA filter parameter:", 0.0, 1.0, 0.1, 0.01)
    bankroll = st.number_input("99% VaR dollar amount:", value=100., min_value=1., max_value=5000., step=1.)
    download_button = st.button("Download stocks")
    allocate_button = st.button("Allocate")
    market_regime_button = st.button("Market Regime")
    option_portfolio = st.radio("Option portfolio", ("Off", "On"))
    market_rates = st.radio("Rates", ("Market", "RH"))
    beta_hedge = st.radio("Beta Hedge", ("Off", "On"))
    is_beta_hedge = beta_hedge == "On"
    is_option_portfolio = option_portfolio == "On"
    is_market_rate = market_rates == "Market"
    expiry_index = st.number_input("Expiry index", 0, 30, 0, 1)
    rate_lb = st.slider("LB for rate search", -20., 0., -1., 0.01)
    rate_ub = st.slider("UB for rate search", 0., 20., 1., 0.01)
    iv_ub = st.slider("UB for IV search", 0.01, 100., 0.3, 0.01)


    if market_regime_button and not is_beta_hedge:
        # Reset to market regime tickers
        symbols = ["SPY", "TLT", "SHY", "VXX"]
        st.session_state.historical_data, st.session_state.timescale = download_data(symbols)
        st.write("Market regime data downloaded successfully.")

    if download_button:
        st.session_state.historical_data, st.session_state.timescale = download_data(symbols)
        st.write("Data downloaded successfully.")

    if allocate_button or market_regime_button:
        st.session_state.historical_data = update_with_quotes(st.session_state.historical_data)

        st.write("## Most recent quotes")
        st.dataframe(st.session_state.quotes)

        if is_beta_hedge:
            beta_symbols = ["SPY"] + symbols + ["VXX"]
            st.session_state.historical_data, st.session_state.timescale = download_data(beta_symbols)
            symbols = beta_symbols

        w, g, mu, sigma = compute_allocations(st.session_state.historical_data, gbm, ema_filter,
                                              st.session_state.timescale, beta_hedge=is_beta_hedge)

        if market_regime_button:
            bull_market = ["SPY"]
            bull_allocations = sum([w[i] for i, asset in enumerate(symbols) if asset in bull_market])
            market_status = "Bull Market" if bull_allocations > 0.5 else "Bear Market"
            st.markdown(f"### Market Regime: **{market_status}**")

        st.write("## EWMA-GBM Allocations")

        if is_beta_hedge:
            display_assets = [asset for asset in symbols if asset != "SPY"]
        else:
            display_assets = symbols

        allocations = {asset: f"{w[i]:.2%}" for i, asset in enumerate(display_assets) if np.abs(w[i]) > 0.001}

        VaR = norm.ppf(0.001, loc=(mu - 0.5 * sigma ** 2) * st.session_state.timescale,
                       scale=sigma * np.sqrt(st.session_state.timescale))

        total = -bankroll / VaR

        dollar_amounts = {asset: round(total * w[i], 2) for i, asset in enumerate(display_assets) if
                          np.abs(w[i]) > 0.001}

        # Combine allocations and dollar amounts into a single dataframe
        allocation_output = {
            'Asset': list(allocations.keys()),
            'Allocation': list(allocations.values()),
            'Dollar Amount': [dollar_amounts[asset] for asset in allocations.keys()]
        }

        allocation_output = pd.DataFrame(allocation_output)

        st.table(allocation_output)
        st.write("## Stats")
        metric_data = {
            "Metric": ["Optimal growth rate", "Annual Drift", "Annual Volatility", "99.9% Daily Value at Risk"],
            "Value": [round(g, 6), round(mu, 4), round(sigma, 4), round(VaR, 4)]
        }

        metric_data = pd.DataFrame(metric_data)
        # display the table
        st.table(metric_data)

        if is_option_portfolio:
            dominant_asset_index = np.argmax(w)
            mu_ema = gbm.drift[dominant_asset_index]
            sigma_ema = np.sqrt(np.diagonal(gbm.Sigma)[dominant_asset_index])
            dominant_asset = display_assets[dominant_asset_index]
            st.write(f"### Optimal Option Strategy for {dominant_asset}")

            option_strategy, max_growth = optimal_option_strategy(dominant_asset,
                                                                  mu_ema,
                                                                  sigma_ema,
                                                                  rf_rate,
                                                                  expiration_date_index=expiry_index,
                                                                  use_market_ivs=is_market_rate)
            st.write(f"Expected max growth: {max_growth}")
            st.write(f"Current EMA drift = {mu_ema}")
            st.write(f"Current EMA volatility = {sigma_ema}")
            st.table(option_strategy)
