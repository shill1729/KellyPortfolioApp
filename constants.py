import numpy as np

rh_apr = 0.05
rf_rate = np.log(1 + rh_apr)
daily_apr = np.exp(rf_rate / 252) - 1