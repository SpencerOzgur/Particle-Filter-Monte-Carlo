import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

def mle_estimate(x:np.array, y:np.array) -> dict:
    def neg_log_likelihood(params):
        phi, alpha, log_sigma2, log_tau2 = params
        sigma2, tau2 = np.exp(log_sigma2), np.exp(log_tau2)

        var_x0 = sigma2 / (1 - phi **2)
        ll = norm.logpdf(x[0], 0, np.sqrt(var_x0))

        ll += np.sum(norm.logpdf(x[1:], phi * x[:-1], np.sqrt(sigma2)))
        ll += np.sum(norm.logpdf(y, alpha * x, np.sqrt(tau2)))
        return -ll

    res = minimize(fun=neg_log_likelihood,
                   x0=np.array([0.5, 0.5, np.log(np.var(y)), np.log(np.var(y))]),
                   method='L-BFGS-B',
                   bounds=[(-0.999, 0.999), (None, None), (None, None), (None, None)])
    phi, alpha, log_sigma2, log_tau2 = res.x
    return {
        'phi': phi,
        'alpha': alpha,
        'sigma2': np.exp(log_sigma2),
        'tau2': np.exp(log_tau2),
        'success': res.success,
        'nll': res.fun
    }

def recover_gap_mle(params: dict, y_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phi   = params['phi']
    alpha = params['alpha']
    sigma = np.sqrt(params['sigma2'])
    tau   = np.sqrt(params['tau2'])

    T = len(y_obs)
    gap     = 0.0
    gap_var = sigma**2 / (1 - phi**2)

    gap_estimates     = np.zeros(T)
    gap_var_estimates = np.zeros(T)

    for t in range(1, T):
        # Predict
        gap_hat     = phi * gap
        gap_var_hat = phi**2 * gap_var + sigma**2

        # Update
        y_err       = y_obs[t] - alpha * gap_hat
        y_var       = alpha**2 * gap_var_hat + tau**2
        kalman_gain = (gap_var_hat * alpha) / y_var

        gap     = gap_hat + kalman_gain * y_err
        gap_var = (1 - kalman_gain * alpha) * gap_var_hat

        gap_estimates[t]     = gap
        gap_var_estimates[t] = gap_var

    return gap_estimates, gap_var_estimates