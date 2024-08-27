# Newtonian Planet Orbit Modeling using MCMC

This project models the orbits of Newtonian planets using the Markov Chain Monte Carlo (MCMC) method. The implementation uses Python, NumPy for numerical computations, and Matplotlib for data visualization. The code generates a semi-random dataset of test trials informed by the planetary orbit model, enabling comprehensive model testing and verification under various orbital scenarios.

## Features
- **Custom MCMC Simulation**: Implemented a custom MCMC simulation to analyze data with a Stochastic Volatility model.
- **Data Generation**: Generates synthetic data for a linear model representing planetary orbits.
- **Visualization**: Utilizes Matplotlib for plotting data and results.
- **Statistical Analysis**: Uses NumPy for numerical operations and statistical analysis.

## Requirements
- Python 3.x
- NumPy
- Matplotlib
- SciPy
- Seaborn

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/rishabh-git10/newtonian-planet-orbit-mcmc.git
    cd newtonian-planet-orbit-mcmc
    ```

2. Install the required packages:
    ```bash
    pip install numpy matplotlib scipy seaborn
    ```

## Usage
1. Run the script:
    ```bash
    python main.py
    ```

2. The script will generate synthetic data, run the MCMC simulation, and plot the results.

## Code Explanation
The main steps of the code are as follows:

1. **Importing Libraries**:
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import seaborn; seaborn.set_theme()
    ```

2. **Generating Synthetic Data**:
    ```python
    def make_data(intercept, slope, N=20, dy=2, rseed=42):
        rand = np.random.RandomState(rseed)
        x = 100 * rand.rand(N)
        y = intercept + slope * x
        y += dy * rand.randn(N)
        return x, y, dy * np.ones_like(x)

    theta_true = [2, 0.5]
    x, y, dy = make_data(theta_true[0], theta_true[1])
    plt.errorbar(x, y, dy, fmt='o')
    ```

3. **Defining the Model and Likelihood Functions**:
    ```python
    def model(theta, x):
        return theta[0] + theta[1] * x

    def ln_likelihood(theta, x, y, dy):
        y_model = model(theta, x)
        return -0.5 * np.sum(np.log(2 * np.pi * dy ** 2) + (y - y_model) ** 2 / dy ** 2)
    ```

4. **Defining the Prior and Posterior Functions**:
    ```python
    def ln_prior(theta):
        if np.all(np.abs(theta[1]) < 100):
            return 0
        return -np.inf

    def ln_posterior(theta, x, y, dy):
        return ln_prior(theta) + ln_likelihood(theta, x, y, dy)
    ```

5. **Running the MCMC Simulation**:
    ```python
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
    ```

6. **Running and Plotting the MCMC Results**:
    ```python
    ndim = 2
    nsteps = 10000
    theta0 = [1, 0]
    stepsize = 0.1
    args = (x, y, dy)
    chain, n_accept = run_mcmc(ln_posterior, nsteps, ndim, theta0, stepsize, args)

    fig, ax = plt.subplots(2)
    ax[0].plot(chain[:, 0])
    ax[1].plot(chain[:, 1])

    # Fresh chain
    chain, n_accept = run_mcmc(ln_posterior, 500000, ndim, chain[-1], 0.1, args)

    fig, ax = plt.subplots(2)
    ax[0].plot(chain[:, 0])
    ax[1].plot(chain[:, 1])

    fig, ax = plt.subplots(2)
    ax[0].hist(chain[:, 0], bins=250, alpha=0.5, density=True)
    ax[1].hist(chain[:, 1], bins=250, alpha=0.5, density=True)

    plt.show()
    ```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
- This project uses the MCMC method to model planetary orbits.
- Inspired by various resources on Bayesian statistics and MCMC simulations.

Feel free to contribute to this project by opening issues or submitting pull requests. Happy coding!