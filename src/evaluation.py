import os
import numpy as np
import pandas as pd

def compute_rmse(y_true, y_pred):
    n_sims = y_true.shape[0]
    rmse = np.zeros(n_sims)

    for i in range(n_sims):
        rmse[i] = np.sqrt(np.mean((y_pred[i] - y_true[i]) ** 2))

    return np.mean(rmse)

def export_rmse(x_true: np.ndarray, results: dict[str, np.ndarray], filename: str = "rmse_results.csv") -> str:
    rmse_dict = {method: compute_rmse(x_true, preds) for method, preds in results.items()}

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    filepath = os.path.join(results_dir, filename)
    pd.DataFrame([rmse_dict]).to_csv(filepath, index=False)

    return filepath

from scipy.stats import ttest_1samp
import pandas as pd
import numpy as np
import os


def test_for_bias(true_params: dict, pred_params: dict[str, np.ndarray]) -> pd.DataFrame:
    param_names = ['phi', 'alpha', 'sigma', 'tau']
    param_idx   = {p: i for i, p in enumerate(param_names)}

    results = []

    for method, estimates in pred_params.items():
        for param in param_names:
            idx         = param_idx[param]
            est         = estimates[:, idx]
            true_val    = true_params[param]

            bias        = np.mean(est) - true_val
            stat, p_val = ttest_1samp(est, popmean=true_val)

            unbiased_95 = p_val > 0.05
            unbiased_99 = p_val > 0.01

            print(
                f"{method} | {param}: "
                f"bias={bias:.4f}, p={p_val:.4f} | "
                f"95% unbiased: {unbiased_95} | "
                f"99% unbiased: {unbiased_99}"
            )

            results.append({
                'method':       method,
                'param':        param,
                'true':         true_val,
                'mean_est':     np.mean(est),
                'bias':         bias,
                'stat':         stat,
                'p_value':      p_val,
                'unbiased_95':  unbiased_95,
                'unbiased_99':  unbiased_99
            })

    df = pd.DataFrame(results)

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, 'bias_tests.csv')
    df.to_csv(filepath, index=False)
    print(f"\nSaved to {filepath}")

    return df