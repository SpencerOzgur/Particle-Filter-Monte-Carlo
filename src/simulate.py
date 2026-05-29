import numpy as np

def simulate(phi: float, alpha: float,
             sigma: float, tau: float,
             T: int, n_sims: int) -> tuple[np.ndarray, np.ndarray]:
    x_sim = np.zeros((n_sims, T))
    y_sim = np.zeros((n_sims, T))

    for i in range(n_sims):
        x = np.zeros(T)
        y = np.zeros(T)

        x[0] = np.normal.random(0, sigma / np.sqrt(1 - phi**2))
        y[0] = alpha * x[0] + np.random.normal(0, tau)

        for t in range(1, T):
            x[t] = phi * x[t-1] + np.random.normal(0, sigma)
            y[t] = alpha * x[t] + np.random.normal(0, tau)

        x_sim[i] = x
        y_sim[i] = y

    return x_sim, y_sim