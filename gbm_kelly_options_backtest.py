# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from black_scholes import random_gbm, bs_price, bs_delta, bs_gamma
# from optimal_mispriced_option import compute_optimal_option_portfolio
#
#
# def implied_volatility_smile(K, S0, base_vol=0.2):
#     return base_vol * (1 + 0.2 * ((K / S0 - 1) ** 2))
#
#
# def generate_option_data(S, strikes, tte, r, base_vol, S0):
#     data = {'strike': [], 'mid': [], 'delta': [], 'gamma': [], 'impliedVolatility': [], 'option_type': []}
#     for K in strikes:
#         implied_vol = implied_volatility_smile(K, S0, base_vol)
#         for option_type in ['call', 'put']:
#             mid = bs_price(S, K, tte, r, implied_vol, option_type)
#             delta = bs_delta(S, K, tte, r, implied_vol, option_type)
#             gamma = bs_gamma(S, K, tte, r, implied_vol)
#
#             data['strike'].append(K)
#             data['mid'].append(mid)
#             data['delta'].append(delta)
#             data['gamma'].append(gamma)
#             data['impliedVolatility'].append(implied_vol)
#             data['option_type'].append(option_type)
#
#     return pd.DataFrame(data)
#
#
# def adjust_strikes(S, base_strikes, base_S0):
#     percentage_changes = [(K - base_S0) / base_S0 for K in base_strikes]
#     adjusted_strikes = [S * (1 + p) for p in percentage_changes]
#     return adjusted_strikes
#
#
# def simulate_backtest(mu, sigma, r, S0, T, steps, num_simulations, base_strikes, option_maturity, base_vol, initial_portfolio_value=1.0, threshold=1e-5):
#     dt = T / steps
#     results = []
#
#     for sim in range(num_simulations):
#         stock_prices = random_gbm(S0, T, mu, sigma, n=steps)
#         portfolio_value = initial_portfolio_value
#         portfolio_values = [portfolio_value]
#         current_strikes = base_strikes
#
#         for i in range(steps):
#             S = stock_prices[i]
#             tte = option_maturity - (i % (steps // (T // option_maturity))) * dt  # rolling expiration
#             if tte <= 0:
#                 tte = option_maturity  # reset time to expiry
#                 current_strikes = adjust_strikes(S, base_strikes, S0)  # adjust strikes based on current spot price
#
#             options_data = generate_option_data(S, current_strikes, tte, r, base_vol, S0)
#             optimal_portfolio, max_growth = compute_optimal_option_portfolio(options_data, S, mu, r, sigma)
#
#             alpha = optimal_portfolio
#             delta = options_data['delta'].values
#             gamma = options_data['gamma'].values
#             V = options_data['mid'].values
#             implied_vols = options_data['impliedVolatility'].values
#
#             valid_idx = V > 0
#             delta = delta[valid_idx]
#             gamma = gamma[valid_idx]
#             V = V[valid_idx]
#             alpha = alpha[valid_idx]
#             implied_vols = implied_vols[valid_idx]
#
#             if len(V) == 0:
#                 break
#
#             eta = (mu - r) * S * (delta / V) + 0.5 * S ** 2 * (sigma ** 2 - implied_vols ** 2) * gamma / V
#             beta0 = 0  # Assuming portfolio consists only of options and bonds
#             a = r + beta0 * (mu - r) + np.dot(eta, alpha)
#             b = beta0 * sigma + sigma * S * np.dot(delta / V, alpha)
#
#             portfolio_value += portfolio_value * a * dt + portfolio_value * b * np.sqrt(dt) * np.random.normal()
#             portfolio_values.append(portfolio_value)
#
#         results.append(portfolio_values)
#
#     results = np.array(results)
#     return results
#
#
# def analyze_results(portfolio_values):
#     final_values = portfolio_values[:, -1]
#     mean_final_value = np.mean(final_values)
#     std_final_value = np.std(final_values)
#
#     log_returns = np.log(portfolio_values[:, 1:] / portfolio_values[:, :-1])
#     mean_daily_log_return = np.mean(log_returns)
#     std_daily_log_return = np.std(log_returns)
#
#     return mean_final_value, std_final_value, mean_daily_log_return, std_daily_log_return
#
#
# if __name__ == "__main__":
#     mu = 0.1  # Example expected return of the stock
#     sigma = 0.201  # Example true volatility
#     r = 0.05  # Example risk-free rate
#     S0 = 150  # Example initial stock price
#     T = 30 / 252  # 1 year
#     steps = 60  # Daily steps
#     num_simulations = 10
#     base_strikes = [140, 145, 150, 155, 160]  # Example strikes
#     option_maturity = 5 / 252  # Example 1 month
#     base_vol = 0.2  # Base volatility for implied volatility smile
#     initial_portfolio_value = 10000  # Initial portfolio value
#
#     portfolio_values = simulate_backtest(mu, sigma, r, S0, T, steps, num_simulations, base_strikes, option_maturity,
#                                          base_vol, initial_portfolio_value)
#
#     mean_final_value, std_final_value, mean_daily_log_return, std_daily_log_return = analyze_results(portfolio_values)
#
#     print(f"Mean final portfolio value: {mean_final_value}")
#     print(f"Standard deviation of final portfolio value: {std_final_value}")
#     print(f"Mean daily log return: {mean_daily_log_return}")
#     print(f"Standard deviation of daily log return: {std_daily_log_return}")
#
#     # Plotting the portfolio value over time for all simulations
#     for i in range(num_simulations):
#         plt.plot(portfolio_values[i])
#     plt.xlabel('Days')
#     plt.ylabel('Portfolio Value')
#     plt.title('Portfolio Value Over Time')
#     plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from black_scholes import random_gbm, bs_price, bs_delta, bs_gamma
from optimal_mispriced_option import compute_optimal_option_portfolio


def implied_volatility_smile(K, S0, base_vol=0.2):
    return base_vol * (1 + 0.2 * ((K / S0 - 1) ** 2))


def generate_option_data(S, strikes, tte, r, base_vol, S0):
    data = {'strike': [], 'mid': [], 'delta': [], 'gamma': [], 'impliedVolatility': [], 'option_type': []}
    for K in strikes:
        implied_vol = implied_volatility_smile(K, S0, base_vol)
        for option_type in ['call', 'put']:
            mid = bs_price(S, K, tte, r, implied_vol, option_type)
            delta = bs_delta(S, K, tte, r, implied_vol, option_type)
            gamma = bs_gamma(S, K, tte, r, implied_vol)

            data['strike'].append(K)
            data['mid'].append(mid)
            data['delta'].append(delta)
            data['gamma'].append(gamma)
            data['impliedVolatility'].append(implied_vol)
            data['option_type'].append(option_type)

    return pd.DataFrame(data)


def adjust_strikes(S, base_strikes, base_S0):
    percentage_changes = [(K - base_S0) / base_S0 for K in base_strikes]
    adjusted_strikes = [S * (1 + p) for p in percentage_changes]
    return adjusted_strikes


def simulate_backtest(mu, sigma, r, S0, T, steps, num_simulations, base_strikes, option_maturity, base_vol, initial_portfolio_value=1.0, threshold=1e-5):
    dt = T / steps
    results = []

    for sim in range(num_simulations):
        stock_prices = random_gbm(S0, T, mu, sigma, n=steps)
        portfolio_value = initial_portfolio_value
        portfolio_values = [portfolio_value]
        current_strikes = base_strikes

        for i in range(steps):
            S = stock_prices[i]
            tte = option_maturity - (i % (steps // (T // option_maturity))) * dt  # rolling expiration
            if tte <= 0:
                tte = option_maturity  # reset time to expiry
                current_strikes = adjust_strikes(S, base_strikes, S0)  # adjust strikes based on current spot price
                # Close out old options
                options_data = generate_option_data(S, current_strikes, tte, r, base_vol, S0)
                old_portfolio_value = np.sum(options_data['mid'].values * alpha)  # get old portfolio value
                portfolio_value += old_portfolio_value  # sell old options and add to portfolio value

            options_data = generate_option_data(S, current_strikes, tte, r, base_vol, S0)
            optimal_portfolio, max_growth = compute_optimal_option_portfolio(options_data, S, mu, r, sigma)

            alpha = optimal_portfolio
            delta = options_data['delta'].values
            gamma = options_data['gamma'].values
            V = options_data['mid'].values
            implied_vols = options_data['impliedVolatility'].values

            valid_idx = V > 0
            delta = delta[valid_idx]
            gamma = gamma[valid_idx]
            V = V[valid_idx]
            alpha = alpha[valid_idx]
            implied_vols = implied_vols[valid_idx]

            if len(V) == 0:
                break

            eta = (mu - r) * S * (delta / V) + 0.5 * S ** 2 * (sigma ** 2 - implied_vols ** 2) * gamma / V
            beta0 = 0  # Assuming portfolio consists only of options and bonds
            a = r + beta0 * (mu - r) + np.dot(eta, alpha)
            b = beta0 * sigma + sigma * S * np.dot(delta / V, alpha)

            portfolio_value += portfolio_value * a * dt + portfolio_value * b * np.sqrt(dt) * np.random.normal()
            portfolio_values.append(portfolio_value)

        results.append(portfolio_values)

    results = np.array(results)
    return results


def analyze_results(portfolio_values):
    final_values = portfolio_values[:, -1]
    mean_final_value = np.mean(final_values)
    std_final_value = np.std(final_values)

    log_returns = np.log(portfolio_values[:, 1:] / portfolio_values[:, :-1])
    mean_daily_log_return = np.mean(log_returns)
    std_daily_log_return = np.std(log_returns)

    return mean_final_value, std_final_value, mean_daily_log_return, std_daily_log_return


if __name__ == "__main__":
    mu = 0.1  # Example expected return of the stock
    sigma = 0.201  # Example true volatility
    r = 0.05  # Example risk-free rate
    S0 = 150  # Example initial stock price
    T = 30 / 252  # 1 year
    steps = 60  # Daily steps
    num_simulations = 10
    base_strikes = [140, 145, 150, 155, 160]  # Example strikes
    option_maturity = 5 / 252  # Example 1 month
    base_vol = 0.2  # Base volatility for implied volatility smile
    initial_portfolio_value = 10000  # Initial portfolio value

    portfolio_values = simulate_backtest(mu, sigma, r, S0, T, steps, num_simulations, base_strikes, option_maturity,
                                         base_vol, initial_portfolio_value)

    mean_final_value, std_final_value, mean_daily_log_return, std_daily_log_return = analyze_results(portfolio_values)

    print(f"Mean final portfolio value: {mean_final_value}")
    print(f"Standard deviation of final portfolio value: {std_final_value}")
    print(f"Mean daily log return: {mean_daily_log_return}")
    print(f"Standard deviation of daily log return: {std_daily_log_return}")

    # Plotting the portfolio value over time for all simulations
    for i in range(num_simulations):
        plt.plot(portfolio_values[i])
    plt.xlabel('Days')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.show()
