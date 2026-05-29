from data import generate_data
from simulate import simulate
from evaluation import compute_rmse
import numpy as np

if __name__ == '__main__':
    np.random.seed(42)
    T = 100
    n_sims= 3

    x_true, y_true = generate_data(n_sims=3)
    y_pred= simulate(T=T, n_sims=n_sims, x_true=x_true, y_true=y_true)

    for method, preds in y_pred.items():
        print(f"{method}: {compute_rmse(x_true, preds):.4f}")