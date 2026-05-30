from data import generate_data
from simulate import simulate
from evaluation import export_rmse, test_for_bias
from plotting import plot_gap_estimates
import numpy as np

if __name__ == '__main__':
    np.random.seed(42)
    T = 100
    n_sims = 5
    phi = 0.9
    alpha = 1.5
    sigma2 = 0.1
    tau2 = 0.5

    true_params = {'phi': phi, 'alpha': alpha, 'sigma': np.sqrt(sigma2), 'tau': np.sqrt(tau2)}

    x_true, y_true = generate_data(T=T, n_sims=n_sims, phi=phi, alpha=alpha, sigma2=sigma2, tau2=tau2)

    results = simulate(T=T, n_sims=n_sims, x_true=x_true, y_true=y_true)
    y_pred = results['gaps']
    params_pred = results['params']

    rmse_path = export_rmse(x_true, y_pred)
    print(f"RMSE saved to {rmse_path}")

    plot_path = plot_gap_estimates(x_true, y_pred, sim_idx=0)
    print(f"Plot saved to {plot_path}")

    bias_df = test_for_bias(true_params, params_pred)