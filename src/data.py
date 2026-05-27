import numpy as np

def generate_data(N:int = 500, phi:float = 0.9,
                  alpha:float = 1.5, sigma2:float = 0.1,
                  tau2:float = 0.5, seed:int = 42) -> tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)
    x = np.zeros(N)
    y = np.zeros(N)

    x[0] = rng.normal(0, np.sqrt(sigma2 / (1 - phi**2)))
    for i in range(1, N):
        x[i] = phi * x[i - 1] + rng.normal(0, np.sqrt(sigma2))

    y = alpha * x + rng.normal(0, np.sqrt(tau2), size=N)
    return x, y