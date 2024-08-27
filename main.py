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

plt.errorbar(x, y, dy, fmt='o')

def model(theta, x):
    return theta[0] + theta[1] * x

def ln_likelihood(theta, x, y, dy):
    y_model = model(theta, x)
    return -0.5 * np.sum(np.log(2 * np.pi * dy ** 2) + (y - y_model) ** 2 / dy ** 2)

def ln_prior(theta):
    if np.all(np.abs(theta[1]) < 100):
        return 0
    return -np.inf

def ln_posterior(theta, x, y, dy):
    return ln_prior(theta) + ln_likelihood(theta, x, y, dy) 

