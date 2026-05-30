import numpy as np
from mle import mle_estimate, recover_gap_mle
from kalman import recover_gap_kf, get_kf_params
from particle import get_pf_params, recover_gap_pf

def simulate(T: int, n_sims: int, x_true: np.ndarray, y_true: np.ndarray) -> dict[str, np.ndarray]:
    mle_point = np.zeros((n_sims, T))
    kf_point = np.zeros((n_sims, T))
    pf_point = np.zeros((n_sims, T))

    mle_var = np.zeros((n_sims, T))
    kf_var = np.zeros((n_sims, T))
    pf_var = np.zeros((n_sims, T))

    param_keys = ['phi', 'alpha', 'sigma2', 'tau2']
    mle_params_all = np.zeros((n_sims, 4))
    kf_params_all = np.zeros((n_sims, 4))
    pf_params_all = np.zeros((n_sims, 4))

    for i in range(n_sims):
        mle_params = mle_estimate(x_true[i], y_true[i])
        kf_params = get_kf_params(y_true[i])
        pf_params = get_pf_params(y_true[i])

        mle_point[i], mle_var[i] = recover_gap_mle(mle_params, y_true[i])
        kf_point[i], kf_var[i] = recover_gap_kf(kf_params, y_true[i])
        pf_point[i], pf_var[i] = recover_gap_pf(pf_params, y_true[i])

        mle_params_all[i] = [mle_params[k] for k in param_keys]
        kf_params_all[i] = kf_params
        pf_params_all[i] = pf_params

    return {
        'gaps': {'MLE': mle_sim, 'KF': kf_sim, 'PF': pf_sim},
        'params': {'MLE': mle_params_all, 'KF': kf_params_all, 'PF': pf_params_all}
    }
