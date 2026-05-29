import os
import numpy as np
import matplotlib.pyplot as plt


def plot_gap_estimates(x_true: np.ndarray, results: dict[str, np.ndarray], sim_idx: int = 0) -> str:
    plot_dir = os.path.join(os.path.dirname(__file__), '..', 'plotting')
    os.makedirs(plot_dir, exist_ok=True)

    T = x_true.shape[1]
    t = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for ax, (method, preds) in zip(axes, results.items()):
        ax.plot(t, x_true[sim_idx], label='True Gap', color='black', linewidth=1.5)
        ax.plot(t, preds[sim_idx], label=f'{method} Estimate', linestyle='--', linewidth=1.5)
        ax.set_title(method)
        ax.legend()
        ax.set_ylabel('Gap')

    axes[-1].set_xlabel('Time')
    fig.suptitle(f'Output Gap: True vs Estimated (Simulation {sim_idx})', fontsize=14)
    plt.tight_layout()

    filepath = os.path.join(plot_dir, f'gap_estimates_sim{sim_idx}.png')
    plt.savefig(filepath, dpi=150)
    plt.close()

    return filepath