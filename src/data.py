import numpy as np

def generate_data(T:int = 100, n_sims=100,
                  phi:float = 0.9, alpha:float = 1.5,
                  sigma2:float = 0.1, tau2:float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    x_sim = np.zeros((n_sims, T))
    y_sim = np.zeros((n_sims, T))

    for i in range(n_sims):
        x = np.zeros(T)

        x[0] = np.random.normal(0, np.sqrt(sigma2 / (1 - phi**2)))
        for t in range(1, T):
            x[t] = phi * x[t - 1] + np.random.normal(0, np.sqrt(sigma2))

        y = alpha * x + np.random.normal(0, np.sqrt(tau2), size=T)
        x_sim[i] = x
        y_sim[i] = y
    return x_sim, y_sim

