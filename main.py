import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn; seaborn.set_theme()

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

def run_mcmc(ln_posterior, nsteps, ndim, theta0, stepsize, args=()):
    chain = np.zeros((nsteps, ndim))
    chain[0] = theta0
    log_prob = ln_posterior(theta0, *args)
    n_accept = 0
    for i in range(1, nsteps):
        theta_new = chain[i - 1] + stepsize * np.random.randn(ndim)
        log_prob_new = ln_posterior(theta_new, *args)
        if (log_prob_new - log_prob) > np.log(np.random.rand()):
            chain[i] = theta_new
            log_prob = log_prob_new
            n_accept += 1
        else:
            chain[i] = chain[i - 1]
    return chain, n_accept

ndim = 2
nsteps = 10000
theta0 = [1, 0]
stepsize = 0.1
args = (x, y, dy)
chain, n_accept = run_mcmc(ln_posterior, nsteps, ndim, theta0, stepsize, args)

fig, ax = plt.subplots(2)
ax[0].plot(chain[:, 0])
ax[1].plot(chain[:, 1])

#Fresh chain
chain, n_accept = run_mcmc(ln_posterior, 500000, ndim, chain[-1], 0.1, args)

fig, ax = plt.subplots(2)
ax[0].plot(chain[:, 0])
ax[1].plot(chain[:, 1])

fig, ax = plt.subplots(2)
ax[0].hist(chain[:, 0], bins=250, alpha=0.5, density=True)
ax[1].hist(chain[:, 1], bins=250, alpha=0.5, density=True)

plt.show()