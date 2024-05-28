import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import norm


# Black-Scholes formulas
def bs_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid 'option_type'; only 'put' or 'call' is allowed")
    return price


def bs_delta(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        delta = norm.cdf(d1)
    elif option_type == 'put':
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError("Invalid 'option_type'; only 'put' or 'call' is allowed")
    return delta


def bs_gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma


# Function to compute the implied risk-free rate
def implied_r(S, K, T, market_price, sigma, option_type='call', a=-50., b=50.):
    def objective(r):
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    sol = root_scalar(objective, bracket=[a, b], method='brentq')
    return sol.root


# Function to compute the implied risk-free rate
def implied_iv(S, K, T, market_price, r, option_type='call', b=5.):
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    sol = root_scalar(objective, bracket=[0.0001, b], method='brentq')
    return sol.root
