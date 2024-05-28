import numpy as np
from sklearn.linear_model import LinearRegression
from optport import mv_solver
from constants import daily_apr, rf_rate


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