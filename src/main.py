from data import generate_data
from mle import mle_estimate
import numpy as np

if __name__ == '__main__':
    np.random.seed(42)

    x_true, y_true = generate_data()
