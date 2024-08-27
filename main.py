import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn; seaborn.set()

def make_data(intercept, slope, N=20, dy=2, rseed=42):
    rand = np.random.RandomState(rseed)
    x = 100 * rand.rand(N)
    y = intercept + slope * x
    y += dy * rand.randn(N)
    return x, y, dy * np.ones_like(x)

theta_true = [2, 0.5]
x, y, dy = make_data(theta_true[0], theta_true[1])

