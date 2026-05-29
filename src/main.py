from data import generate_data
from simulate import simulate
from evaluation import compute_rmse, export_rmse
from plotting import plot_gap_estimates
import numpy as np

if __name__ == '__main__':
    np.random.seed(42)
    T = 100
    n_sims = 5

    x_true, y_true = generate_data(T=T, n_sims=n_sims)
    y_pred = simulate(T=T, n_sims=n_sims, x_true=x_true, y_true=y_true)

    rmse_path = export_rmse(x_true, y_pred)
    print(f"RMSE saved to {rmse_path}")

    plot_path = plot_gap_estimates(x_true, y_pred, sim_idx=0)
    print(f"Plot saved to {plot_path}")