import numpy as np
from scipy.optimize import minimize
from option_data_processing import get_option_chain, get_quote
from black_scholes import bs_price, bs_delta, bs_gamma, implied_r, implied_iv


# Compute eta vector
def compute_eta(mu, r, S, delta, V, sigma, implied_vol, gamma):
    return (mu - r) * S * (delta / V) + 0.5 * S ** 2 * (sigma ** 2 - implied_vol ** 2) * gamma / V


# Compute sigma matrix
def compute_sigma_matrix(delta, V):
    delta_V = delta / V
    return np.outer(delta_V, delta_V)


# Growth function to maximize
def growth_function(alpha, eta, sigma, S, delta, V):
    term1 = np.dot(eta, alpha)
    sigma_matrix = compute_sigma_matrix(delta, V)
    term2 = 0.5 * sigma ** 2 * S ** 2 * np.dot(alpha.T, np.dot(sigma_matrix, alpha))
    return term1 - term2


# Constraint for portfolio weights
def constraint(alpha):
    return np.sum(alpha) - 1


# Optimize portfolio
def compute_optimal_option_portfolio(options_data, S, mu, r, sigma):
    V = options_data['mid'].values
    delta = options_data['delta'].values
    gamma = options_data['gamma'].values
    implied_vol = options_data['impliedVolatility'].values

    eta = compute_eta(mu, r, S, delta, V, sigma, implied_vol, gamma)

    cons = {'type': 'eq', 'fun': constraint}
    bounds = [(0., 1.)] * len(V)

    initial_alpha = np.ones(len(V)) / len(V)

    result = minimize(lambda alpha: -growth_function(alpha, eta, sigma, S, delta, V),
                      initial_alpha, bounds=bounds, constraints=cons)

    if result.success:
        return result.x, -result.fun
    else:
        raise ValueError("Optimization failed")


def optimal_option_strategy(ticker, mu, sigma, r, expiration_date_index=0, threshold=1e-5, use_market_ivs=True):
    S = get_quote(ticker)
    options_data = get_option_chain(ticker)

    # Work with the specified expiration date
    expiration_date = options_data['expirationDate'].unique()[expiration_date_index]
    options_data_expiry = options_data[options_data['expirationDate'] == expiration_date].copy()
    options_data_expiry = options_data_expiry.head(5)

    # Compute deltas, gammas, and implied risk-free rates for each option
    deltas = []
    gammas = []
    implied_rs = []
    implied_vols_rh = []
    for index, row in options_data_expiry.iterrows():
        K = row['strike']
        T = row['tte']
        market_price = row['mid']
        implied_vol = row['impliedVolatility']
        option_type = row['option_type']
        # Either us market volatilities and find implied rates to discount correctly for the greeks
        if use_market_ivs:
            r_implied = implied_r(S, K, T, market_price, implied_vol, option_type)
            implied_rs.append(r_implied)
        else: # or use the RH APR as the risk-free rate and back out new volatilities
            r_implied = r
            implied_vol = implied_iv(S, K, T, market_price, r, option_type)
            implied_vols_rh.append(implied_vol)

        delta = bs_delta(S, K, T, r_implied, implied_vol, option_type)
        gamma = bs_gamma(S, K, T, r_implied, implied_vol)

        deltas.append(delta)
        gammas.append(gamma)

    options_data_expiry['delta'] = deltas
    options_data_expiry['gamma'] = gammas
    if use_market_ivs:
        options_data_expiry['implied_r'] = implied_rs
    else:
        options_data_expiry["fakeIV"] = np.array(implied_vols_rh)
    optimal_portfolio, max_growth = compute_optimal_option_portfolio(options_data_expiry, S, mu, r, sigma)

    # Add the optimal allocations to the DataFrame
    options_data_expiry['optimal_allocation'] = optimal_portfolio

    # Filter and return only the rows with substantial non-zero allocations
    substantial_allocations = options_data_expiry[options_data_expiry['optimal_allocation'] > threshold]
    if use_market_ivs:
        output = ["expirationDate", "strike", "mid", "lastPrice", "volume",
                                        "openInterest", "impliedVolatility", "delta", "gamma",
                                        "option_type", "implied_r", "optimal_allocation"]
    else:
        output = ["expirationDate", "strike", "mid", "lastPrice", "volume",
                  "openInterest", "impliedVolatility", "fakeIV", "delta", "gamma",
                  "option_type", "optimal_allocation"]

    return (substantial_allocations[output],
            max_growth)


if __name__ == "__main__":
    ticker = 'AAPL'
    mu = 1.146376  # Example expected return of the stock
    sigma = 0.06849864  # Example true volatility
    r = np.log(1+0.05)
    result, max_growth = optimal_option_strategy(ticker, mu, sigma, r)
    print(result.to_string())
    print("Expected max growth = "+str(max_growth))
