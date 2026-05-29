def compute_rmse(y_true, y_pred):
    n_sims = y_true.shape[0]
    rmse = np.zeros(n_sims)

    for i in range(n_sims):
        rmse[i] = np.sqrt(np.mean((y_pred[i] - y_true[i]) ** 2))

    return np.mean(rmse)

