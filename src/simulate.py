import numpy as np
from mle import mle_estimate, recover_gap_mle
from kalman import recover_gap_kf, get_kf_params
from particle import get_pf_params, recover_gap_pf

def simulate(T: int, n_sims: int, x_true: np.ndarray, y_true: np.ndarray) -> dict[str, np.ndarray]:
    mle_sim = np.zeros((n_sims, T))
    kf_sim  = np.zeros((n_sims, T))
    pf_sim  = np.zeros((n_sims, T))

    for i in range(n_sims):
        mle_params = mle_estimate(x_true[i], y_true[i])
        kf_params  = get_kf_params(y_true[i])
        pf_params  = get_pf_params(y_true[i])

        mle_sim[i] = recover_gap_mle(mle_params, y_true[i])[0]
        kf_sim[i] = recover_gap_kf(kf_params, y_true[i])[0]
        pf_sim[i] = recover_gap_pf(pf_params, y_true[i])[0]

    return {'MLE': mle_sim, 'KF': kf_sim, 'PF': pf_sim}