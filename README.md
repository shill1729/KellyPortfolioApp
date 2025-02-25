# KellyPortfolioApp
This is a streamlit app, [KellyPortfolio](https://kellyportfolio.streamlit.app/), for finding the 
allocations $\vec{\alpha}$ for a given set of stocks that maximize
log growth 

$$G_t(\vec{\alpha}) = E(\log X_T^\alpha/X_t^\alpha | F_t)$$

In general, a path dependent GBM can be written in the form,
for a single stock, as the SDE
$$d S_t = \mu(t, S_.) S_t dt + \sigma(t, S_.) S_t dB_t$$
where $\mu(t, s_.)$ and $\sigma(t, s_.)$ are previsible path functionals.

Here, we model the stock 
prices as a multivariate GBM where the drift vector and volatility 
matrix coefficients are estimated using exponentially weighted moving averages (EWMAs). 
The optimization problem to solve is then the quadratic problem

$$\max_{\alpha \in C} \mu^T \alpha - \frac12 \alpha^T \Sigma \alpha$$

where $C$ is the constraint set of positive weights that sum to one, i.e.
a probability simplex. In financial terms, this is the set of non-leveraged
allocations that budget to 100\% of one's wealth. Note the unconstrained
solution is simply

$$\alpha^* = \Sigma^{-1} \mu.$$



The app prints the allocations for stocks with non-zero weight, the
optimal growth value under such allocations, the drift
and volatility of the portfolio, and the daily 99.9\% Value-at-Risk,
i.e. worst-case loss on a given day, which has 0.1\% chance of occurring,
per the fitted normal distribution with mean $\mu^T \alpha-\frac12 \alpha^T \Sigma \alpha$
and variance $\alpha^T \Sigma \alpha$ where $\alpha=\alpha^*$.


# Limitations:
The data-provider is AlphaVantage. This app uses my premium
API key to download data. The limit is 30 stocks per minute.



