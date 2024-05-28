import numpy as np
rng = np.random.default_rng()


def random_gbm(s0, tn, mu, sigma, n=1000):
    x = np.zeros(n+1)
    x[0] = 0
    h = tn/n
    for i in range(n):
        Z = rng.normal(size=1)
        x[i+1] = x[i-1]+(mu-0.5*sigma**2)*h + np.sqrt(h)*Z*sigma
    s = s0*np.exp(x)
    return s

