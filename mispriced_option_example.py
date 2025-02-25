import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
S0 = 100  # Initial stock price
mu = 0.1  # Drift
sigma = 0.2  # True volatility
sigma_iv = 0.25  # Implied volatility
r = 0.  # Risk-free rate
K = 100  # Strike price
T = 1.0  # Time to maturity (years)
dt = 1 / 252  # Daily time step
num_steps = int(T / dt)  # Number of time steps
num_paths = 10  # Single sample path for comparison
x0 = 100  # initial wealth


# Black-Scholes helpers
def d1(S, K, T, r, sigma):
    """Calculate d1 for Black-Scholes."""
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes option price."""
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d1_val - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)


def black_scholes_delta(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes delta."""
    d1_val = d1(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1_val)
    elif option_type == "put":
        return norm.cdf(d1_val) - 1


def black_scholes_gamma(S, K, T, r, sigma):
    """Black-Scholes gamma."""
    d1_val = d1(S, K, T, r, sigma)
    return norm.pdf(d1_val) / (S * sigma * np.sqrt(T))


# Simulate GBM for the stock price
np.random.seed(42)
S = np.zeros(num_steps + 1)
S[0] = S0
dW = np.random.normal(0, np.sqrt(dt), num_steps)

for t in range(1, num_steps + 1):
    S[t] = S[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dW[t - 1])

# Simulate portfolios
X_option = np.zeros(num_steps + 1)
X_mispriced = np.zeros(num_steps + 1)
X_option[0] = X_mispriced[0] = x0  # Initial wealth

for t in range(1, num_steps + 1):
    T_t = T - t * dt  # Time to maturity
    if T_t > 0:
        # Compute option values and Greeks
        v = black_scholes_price(S[t - 1], K, T_t, r, sigma)
        delta = black_scholes_delta(S[t - 1], K, T_t, r, sigma)
        gamma = black_scholes_gamma(S[t - 1], K, T_t, r, sigma)

        # Kelly fractions
        alpha_option = (mu - r) * v / (sigma ** 2 * delta * S[t - 1])
        alpha_mispriced = alpha_option + 0.5 * (sigma ** 2 - sigma_iv ** 2) * gamma * v / (sigma ** 2 * delta ** 2)

        # Update portfolios
        dX_option = (
                (r + alpha_option * ((mu - r) * delta / v + 0.5 * (sigma ** 2 - sigma_iv ** 2) * gamma / v)) * X_option[
            t - 1] * dt
                + alpha_option * sigma * delta * S[t - 1] / v * X_option[t - 1] * dW[t - 1]
        )
        dX_mispriced = (
                (r + alpha_mispriced * ((mu - r) * delta / v + 0.5 * (sigma ** 2 - sigma_iv ** 2) * gamma / v)) *
                X_mispriced[t - 1] * dt
                + alpha_mispriced * sigma * delta * S[t - 1] / v * X_mispriced[t - 1] * dW[t - 1]
        )

        X_option[t] = X_option[t - 1] + dX_option
        X_mispriced[t] = X_mispriced[t - 1] + dX_mispriced

# Plot results
time = np.linspace(0, T, num_steps + 1)
plt.figure(figsize=(10, 6))
plt.plot(time, X_option, label="Fairly Priced Option Portfolio")
plt.plot(time, X_mispriced, label="Mispriced Option Portfolio")
plt.xlabel("Time (years)")
plt.ylabel("Portfolio Value")
plt.title("Portfolio Values: Fairly Priced vs Mispriced Options")
plt.legend()
plt.grid()
plt.show()
