import numpy as np
from scipy.optimize import minimize


def obj_kalman_filter(params, gdp_obs):
    phi, theta, sig, tau = params

    # Time observations
    T = len(gdp_obs)

    ##Initialize
    gap = 100
    gap_var = 0.01

    neg_ll = 0

    # Kalman Update
    for t in range(1, T):
        gap_hat = phi * gap
        gap_var_hat = phi ** 2 * gap_var + sig ** 2

        gdp_hat = theta * gap_hat
        gdp_forecast_var = theta ** 2 * gap_var_hat + tau ** 2
        gdp_forecast_err = gdp_obs[t] - gdp_hat

        neg_ll += (
                np.log(gdp_forecast_var)
                + (gdp_forecast_err ** 2) / gdp_forecast_var
        )

        kalman_gain = (gap_var_hat * theta) / gdp_forecast_var

        gap = gap_hat + (kalman_gain * gdp_forecast_err)

        gap_var = (1 - kalman_gain * theta) * gap_var_hat

    return neg_ll


phi_lb = -0.99
phi_ub = 0.99

theta_lb = -5
theta_ub = 5

sig_lb = 1e-4
sig_ub = 100

tau_lb = 1e-4
tau_ub = 100

lb_list = [phi_lb, theta_lb, sig_lb, tau_lb]
ub_list = [phi_ub, theta_ub, sig_ub, tau_ub]


def get_kf_params(gdp_obs):
    # Initial guesses

    phi_0 = 0.5
    theta_0 = 0.5

    sig_0 = np.std(gdp_obs)
    tau_0 = np.std(gdp_obs)

    x0 = np.array([phi_0, theta_0,
                   sig_0, tau_0])

    # Parameter bounds

    phi_lb = -0.99
    phi_ub = 0.99

    theta_lb = -5
    theta_ub = 5

    sig_lb = 1e-4
    sig_ub = 100

    tau_lb = 1e-4
    tau_ub = 100

    lb_list = [phi_lb, theta_lb, sig_lb, tau_lb]
    ub_list = [phi_ub, theta_ub, sig_ub, tau_ub]

    bounds = list(zip(lb_list, ub_list))

    sol = minimize(
        lambda params: obj_kalman_filter(params, gdp_obs),
        bounds=bounds, x0=x0, method='L-BFGS-B'
    )

    return sol.x




def recover_gap_kf(optimal_params, gdp_obs):
    phi, theta, sig, tau = optimal_params

    # Time observations
    T = len(gdp_obs)

    ##Initialize
    gap = 0.0
    gap_var = 0.1

    # Store estimates
    gap_estimates = np.zeros(T)
    gap_var_estimates = np.zeros(T)

    # Kalman Update
    for t in range(1, T):
        gap_hat = phi * gap
        gap_var_hat = phi ** 2 * gap_var + sig ** 2

        gdp_hat = theta * gap_hat
        gdp_forecast_var = theta ** 2 * gap_var_hat + tau ** 2
        gdp_forecast_err = gdp_obs[t] - gdp_hat

        kalman_gain = (gap_var_hat * theta) / gdp_forecast_var

        gap = gap_hat + (kalman_gain * gdp_forecast_err)

        gap_var = (1 - kalman_gain * theta) * gap_var_hat

        gap_estimates[t] = gap
        gap_var_estimates[t] = gap_var

    return gap_estimates, gap_var_estimates

