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


# Utility functions
def fetch_quotes(symbols):
    """Fetch the latest quotes for given symbols using Finnhub."""
    finnhub_client = finnhub.Client(api_key=st.secrets["FINNHUB_KEY"])
    quotes = {}
    for symbol in symbols:
        try:
            quote = finnhub_client.quote(symbol)["c"]
            quotes[symbol] = quote
            time.sleep(1 / len(symbols))  # Throttle API requests
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
    return quotes


def update_with_quotes(prices):
    """Update prices DataFrame with the latest quotes."""
    today = dt.date.today()
    symbols = prices.columns

    quotes = fetch_quotes(symbols)
    if not quotes:
        st.warning("No new quotes fetched.")
        return prices

    new_row = pd.DataFrame([quotes], index=[today])
    st.session_state.quotes = new_row

    prices = prices.combine_first(new_row)
    return prices


@st.cache_data(show_spinner=False)
def download_data(symbols):
    """Download historical data for the given symbols."""
    api = av.av()
    api.log_in(st.secrets["AV_API_KEY"])

    data = api.get_assets(symbols, "daily", None, True, "adjusted_close")
    now = dt.datetime.now().astimezone(dt.timezone(dt.timedelta(hours=-4)))
    is_market_hours = now.weekday() < 5 and dt.time(9, 30) <= now.time() <= dt.time(18, 0)

    if is_market_hours:
        data = update_with_quotes(data)
    else:
        st.write("Previous Close Prices:")
        st.write(data.iloc[-1, :])
    return data, av.timescale("daily", None, "stocks")


# Main Streamlit application
def main():
    """Main function to render Streamlit app."""
    gbm = MultiGbm()
    st.title("Optimal Log Growth Allocations")

    # Input Section
    default_ticker_list = "TSLA, AMC, GME, NKLA, GOTU, NVDA, AAPL, DIS, ROKU, AMZN"
    ticker_list = st.text_input("Enter a comma-separated list of tickers", default_ticker_list)
    symbols = [s.strip() for s in ticker_list.split(",")]

    # Initialize session state
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None

    if 'timescale' not in st.session_state:
        st.session_state.timescale = None

    if 'quotes' not in st.session_state:
        st.session_state.quotes = pd.DataFrame(columns=symbols)

    # Inputs for configurations
    ema_filter = st.slider("Select the EMA filter parameter:", 0.0, 1.0, 0.1, 0.01)
    bankroll = st.number_input("99% VaR dollar amount:", value=100., min_value=1., max_value=5000., step=1.)
    download_button = st.button("Download stocks")
    allocate_button = st.button("Allocate")
    market_regime_button = st.button("Market Regime")
    option_portfolio = st.radio("Option portfolio", ("Off", "On"))
    market_rates = st.radio("Rates", ("Market", "RH"))
    beta_hedge = st.radio("Beta Hedge", ("Off", "On"))
    otm = st.radio("Moneyness:", ("OTM", "ITM"))
    expiry_index = st.number_input("Expiry index", 0, 30, 0, 1)
    rate_lb = st.slider("LB for rate search", -20., 0., -1., 0.01)
    rate_ub = st.slider("UB for rate search", 0., 20., 1., 0.01)
    iv_ub = st.slider("UB for IV search", 0.01, 100., 0.3, 0.01)

    # Button Actions
    if market_regime_button:
        st.session_state.historical_data, st.session_state.timescale = download_data(["SPY", "TLT", "SHY", "VXX"])
        st.write("Market regime data downloaded successfully.")

    if download_button:
        st.session_state.historical_data, st.session_state.timescale = download_data(symbols)
        st.write("Data downloaded successfully.")

    if allocate_button or market_regime_button:
        st.session_state.historical_data = update_with_quotes(st.session_state.historical_data)

        st.write("## Most Recent Quotes")
        st.dataframe(st.session_state.quotes)

        # Extend symbols if beta hedge is on
        if beta_hedge == "On":
            beta_symbols = ["SPY"] + symbols + ["VXX"]
            st.session_state.historical_data, st.session_state.timescale = download_data(beta_symbols)
            symbols = beta_symbols

        # Compute allocations
        w, g, mu, sigma = compute_allocations(
            st.session_state.historical_data, gbm, ema_filter,
            st.session_state.timescale, beta_hedge=(beta_hedge == "On")
        )

        # Display Market Regime
        if market_regime_button:
            bull_allocations = sum([w[i] for i, asset in enumerate(symbols) if asset == "SPY"])
            market_status = "Bull Market" if bull_allocations > 0.5 else "Bear Market"
            st.markdown(f"### Market Regime: **{market_status}**")

        # Display Allocations
        display_assets = [asset for asset in symbols if asset != "SPY"] if beta_hedge == "On" else symbols
        allocations = {asset: f"{w[i]:.2%}" for i, asset in enumerate(display_assets) if np.abs(w[i]) > 0.001}

        VaR = norm.ppf(0.001, loc=(mu - 0.5 * sigma ** 2) * st.session_state.timescale,
                       scale=sigma * np.sqrt(st.session_state.timescale))

        total = -bankroll / VaR

        dollar_amounts = {asset: round(total * w[i], 2) for i, asset in enumerate(display_assets) if
                          np.abs(w[i]) > 0.001}

        allocation_output = pd.DataFrame({
            'Asset': list(allocations.keys()),
            'Allocation': list(allocations.values()),
            'Dollar Amount': [dollar_amounts[asset] for asset in allocations.keys()]
        })

        st.table(allocation_output)
        st.write("## Stats")
        metric_data = pd.DataFrame({
            "Metric": ["Optimal growth rate", "Annual Drift", "Annual Volatility", "99.9% Daily Value at Risk"],
            "Value": [round(g, 6), round(mu, 4), round(sigma, 4), round(VaR, 4)]
        })

        st.table(metric_data)

        if option_portfolio == "On":
            dominant_asset_index = np.argmax(w)
            mu_ema = gbm.drift[dominant_asset_index]
            sigma_ema = np.sqrt(np.diagonal(gbm.Sigma)[dominant_asset_index])
            dominant_asset = display_assets[dominant_asset_index]
            st.write(f"### Optimal Option Strategy for {dominant_asset}")

            option_strategy, max_growth = optimal_option_strategy(
                dominant_asset, mu_ema, sigma_ema, rf_rate,
                expiration_date_index=expiry_index,
                use_market_ivs=(market_rates == "Market"),
                otm=(otm == "OTM"),
                r_lb=rate_lb, r_ub=rate_ub, iv_ub=iv_ub
            )
            st.write(f"Expected max growth: {max_growth}")
            st.write(f"Current EMA drift = {mu_ema}")
            st.write(f"Current EMA volatility = {sigma_ema}")
            st.table(option_strategy)


if __name__ == "__main__":
    main()
