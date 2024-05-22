import numpy as np
import streamlit as st
import datetime as dt
import pandas as pd
import alphavantage.av as av
from scipy.stats import norm
from optport import mv_solver
from sdes import MultiGbm
import finnhub
import time
from sklearn.linear_model import LinearRegression

rh_apr = 0.05
rf_rate = np.log(1 + rh_apr)
daily_apr = np.exp(rf_rate / 252) - 1


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


def compute_allocations(prices, gbm, ema_filter=0.0, timescale=1 / 252, beta_hedge=False):
    log_returns = prices.apply(lambda x: np.diff(np.log(x)))
    arithmetic_returns = log_returns.apply(lambda x: np.exp(x) - 1)

    if beta_hedge:
        symbols = prices.columns
        if "SPY" not in symbols:
            st.error("SPY not in the list of symbols")
            return

        spy_returns = arithmetic_returns["SPY"].values.reshape(-1, 1)
        betas = []
        for symbol in symbols:
            if symbol != "SPY":
                reg = LinearRegression()
                reg.fit(spy_returns - daily_apr, arithmetic_returns[symbol].values - daily_apr)
                betas.append(reg.coef_[0])
        betas = np.array(betas)
        gbm.fit(log_returns.iloc[:, 1:], ema_filter=ema_filter, timescale=timescale)  # fit excluding SPY
        w, g = mv_solver(gbm.drift - rf_rate, gbm.Sigma, betas=betas)
    else:
        gbm.fit(log_returns, ema_filter=ema_filter, timescale=timescale)
        w, g = mv_solver(gbm.drift - rf_rate, gbm.Sigma)

    mu = w.dot(gbm.drift - rf_rate)
    sigma = np.sqrt((w.T.dot(gbm.Sigma)).dot(w))
    return w, g, mu, sigma


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
    beta_hedge = st.radio("Beta Hedge", ("Off", "On"))
    is_beta_hedge = beta_hedge == "On"

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

        # st.write("## EWMA-GBM Allocations")
        # if is_beta_hedge:
        #     display_assets = [asset for asset in symbols if asset != "SPY"]
        # else:
        #     display_assets = symbols
        # allocations = {asset: f"{w[i]:.2%}" for i, asset in enumerate(display_assets) if np.abs(w[i]) > 0.001}
        # st.table(pd.DataFrame(list(allocations.items()), columns=['Asset', 'Allocation']))
        #
        # VaR = norm.ppf(0.001, loc=(mu - 0.5 * sigma ** 2) * st.session_state.timescale,
        #                scale=sigma * np.sqrt(st.session_state.timescale))
        #
        # total = -bankroll / VaR
        # st.write("## Dollar amounts to hold")
        # dollar_amounts = {asset: round(total * w[i], 2) for i, asset in enumerate(display_assets) if
        #                   np.abs(w[i]) > 0.001}
        # st.table(pd.DataFrame(list(dollar_amounts.items()), columns=['Asset', 'Dollar Amount']))
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
