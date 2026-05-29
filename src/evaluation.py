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