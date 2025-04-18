import numpy as np
import streamlit as st
import datetime as dt
import pandas as pd
import alphavantage.av as av
import finnhub
import time
import pandas_market_calendars as mcal
from scipy.stats import norm
from allocator import compute_allocations
from sdes import MultiGbm
from optimal_mispriced_option import optimal_option_strategy
from constants import rf_rate



def get_last_trading_day(today: dt.date) -> dt.date:
    # Example: using NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    # For safety, look up to a week in the past in case 'today' is a Monday or holiday
    schedule = nyse.valid_days(start_date=today - dt.timedelta(days=7), end_date=today)
    if schedule.empty:
        # Fallback: if something's off with the schedule, just return 'today'
        return today
    return schedule[-1].date()  # The last valid trading date in that range


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

    if today in prices.index:
        prices.loc[today] = quotes
    else:
        prices = pd.concat([prices, new_row])

    return prices


# @st.cache_data(show_spinner=False)
# def download_data(symbols):
#     api = av.av()
#     api.log_in(st.secrets["AV_API_KEY"])
#
#     data = api.get_assets(symbols, "daily", None, True, "adjusted_close")
#     now = dt.datetime.now().astimezone(dt.timezone(dt.timedelta(hours=-4)))
#     is_market_hours = now.weekday() < 5 and dt.time(9, 30) <= now.time() <= dt.time(18, 0)
#
#     if is_market_hours:
#         data = update_with_quotes(data)
#     else:
#         st.write("Previous Close Prices:")
#         st.write(data.iloc[-1, :])
#     return data, av.timescale("daily", None, "stocks")
@st.cache_data(show_spinner=False)
def download_data(symbols, last_downloaded_date=None):
    """
    Download historical data from AlphaVantage for the given symbols,
    but only if we do not already have the last trading day's data.
    """
    today = dt.date.today()
    last_trading_day = get_last_trading_day(today)

    # Check if we already have the latest data in session_state
    if last_downloaded_date is not None and last_downloaded_date >= last_trading_day:
        st.write(f"Using cached historical data from {last_downloaded_date}.")
        return st.session_state.historical_data

    # Otherwise, fetch new data
    api = av.av()
    api.log_in(st.secrets["AV_API_KEY"])

    # Fetch fresh historical data
    data = api.get_assets(symbols, "daily", None, True, "adjusted_close")

    # The last date in the DataFrame
    if not data.empty:
        latest_data_date = data.index[-1].date()
    else:
        latest_data_date = today  # fallback if data returned is empty

    st.session_state.last_downloaded_date = latest_data_date
    st.session_state.historical_data = data
    st.write("Previous Close Prices:")
    st.write(data.iloc[-1, :])

    return data, av.timescale("daily", None, "stocks")


if __name__ == "__main__":
    gbm = MultiGbm()
    st.title("Optimal Log Growth Allocations")
    default_ticker_list = "wmt, cost, pg, ko, pep, mo, pm, cl, gis, clx, lmt, noc, gd, rtx, ba"
    ticker_list = st.sidebar.text_input("Enter a comma-separated list of tickers", default_ticker_list)
    symbols = [s.strip().upper() for s in ticker_list.split(",")]

    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None

    if 'timescale' not in st.session_state:
        st.session_state.timescale = None

    if 'quotes' not in st.session_state:
        st.session_state.quotes = pd.DataFrame(columns=symbols)

    ema_filter = st.sidebar.slider("Select the EMA filter parameter:", 0.0, 1.0, 0.1, 0.01)
    bankroll = st.sidebar.number_input("99% VaR dollar amount:", value=100., min_value=1., max_value=5000., step=1.)
    download_button = st.sidebar.button("Download stocks")
    allocate_button = st.sidebar.button("Allocate")
    a = st.sidebar.number_input("Lower bound", min_value=0.01, max_value=10000., value=50.)
    b = st.sidebar.number_input("Upper bound", min_value=0.01, max_value=10000., value=900.)
    market_regime_button = st.sidebar.button("Market Regime")
    option_portfolio = st.sidebar.radio("Option portfolio", ("Off", "On"))
    market_rates = st.sidebar.radio("Rates", ("Market", "RH"))
    beta_hedge = st.sidebar.radio("Beta Hedge", ("Off", "On"))
    otm = st.sidebar.radio("Moneyness:", ("OTM", "ITM"))
    expiry_index = st.sidebar.number_input("Expiry index", 0, 30, 0, 1)
    rate_lb = st.sidebar.slider("LB for rate search", -20., 0., -1., 0.01)
    rate_ub = st.sidebar.slider("UB for rate search", 0., 20., 1., 0.01)
    iv_ub = st.sidebar.slider("UB for IV search", 0.01, 100., 0.3, 0.01)

    if market_regime_button and beta_hedge == "Off":
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

        if beta_hedge == "On":
            beta_symbols = ["SPY"] + symbols + ["VXX"]
            st.session_state.historical_data, st.session_state.timescale = download_data(beta_symbols)
            symbols = beta_symbols
        # TODO: add P_x(T_a < T_b) chance of hitting a before b.
        w, g, mu, sigma, gbm = compute_allocations(st.session_state.historical_data, gbm, ema_filter,
                                              st.session_state.timescale, beta_hedge=beta_hedge == "On")
        dom_asset_index = np.argmax(w)
        dom_asset_drift = gbm.drift[dom_asset_index]
        dom_asset_vol = np.sqrt(gbm.Sigma[dom_asset_index, dom_asset_index])
        gamma = 1-2*dom_asset_drift/dom_asset_vol**2
        spot = st.session_state.historical_data.iloc[-1, dom_asset_index]
        if a > spot:
            a = spot/2
        if b < spot:
            b = 2*spot
        hit_a_before_b = (spot**gamma-b**gamma)/(a**gamma-b**gamma)
        sharpe_ratio = 0.5*((dom_asset_drift-rf_rate)/dom_asset_vol)**2
        vol_drag = (dom_asset_drift-rf_rate)-0.5*dom_asset_vol**2
        dom_asset_kelly = (dom_asset_drift-rf_rate)/dom_asset_vol**2
        if market_regime_button:
            bull_market = ["SPY"]
            bull_allocations = sum([w[i] for i, asset in enumerate(symbols) if asset in bull_market])
            market_status = "Bull Market" if bull_allocations > 0.5 else "Bear Market"
            st.markdown(f"### Market Regime: **{market_status}**")

        st.write("## EWMA-GBM Allocations")

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
            "Metric": ["Constr. optimal growth rate", "Annual Drift", "Annual Volatility", "99.9% Daily Value at Risk",
                       "Chance to hit "+str(a)+" before "+str(b), "Sharpe ratio (unconstrained growth)", "Drift-Vol-Drag", "Dom-Asset KF"],
            "Value": [round(g, 6), round(mu, 4), round(sigma, 4), round(VaR, 4),
                      round(hit_a_before_b, 2), round(sharpe_ratio, 5), round(vol_drag, 4), round(dom_asset_kelly, 4)]
        })

        st.table(metric_data)

        if option_portfolio == "On":
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
                                                                  use_market_ivs=market_rates == "Market",
                                                                  otm=otm == "OTM",
                                                                  r_lb=rate_lb,
                                                                  r_ub=rate_ub,
                                                                  iv_ub=iv_ub)
            st.write(f"Expected max growth: {max_growth}")
            st.write(f"Current EMA drift = {mu_ema}")
            st.write(f"Current EMA volatility = {sigma_ema}")

            st.table(option_strategy)
