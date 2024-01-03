import numpy as np
from torch.utils.data import DataLoader
import os
from .coresetmethod import CoresetMethod
import numpy as np
import copy
import time
from scipy import optimize
from datetime import datetime
from .AproxMVEE import MVEEApprox

# code from https://github.com/muradtuk/Provable-Data-Subset-Selection-For-Efficient-Neural-Network-Training/tree/main

R = 10

def obtainSensitivity(X, w, approxMVEE=False):
    if not approxMVEE:
        return computeSensitivity(X, w)
    else:
        cost_func = lambda x: np.linalg.norm(np.dot(X, x), ord=1)
        mvee = MVEEApprox(X, cost_func, 3)
        ellipsoid, center = mvee.compute_approximated_MVEE()
        U = X.dot(ellipsoid)
        return np.linalg.norm(U, ord=1, axis=1)


def generateCoreset(X, y, sensitivity, sample_size, weights=None, SEED=1):
    if weights is None:
        weights = np.ones((X.shape[0], 1)).flatten()

    # Compute the sum of sensitivities.
    t = np.sum(sensitivity)

    # The probability of a point prob(p_i) = s(p_i) / t
    probability = sensitivity.flatten() / t

    startTime = time.time()

    # initialize new seed
    np.random.seed()

    # Multinomial Distribution
    hist = np.random.choice(np.arange(probability.shape[0]), size=sample_size, replace=False, p=probability.flatten())
    indxs, counts = np.unique(hist, return_counts=True)
    S = X[indxs]
    labels = y[indxs]

    # Compute the weights of each point: w_i = (number of times i is sampled) / (sampleSize * prob(p_i))
    weights = np.asarray(np.multiply(weights[indxs], counts), dtype=float).flatten()

    weights = np.multiply(weights, 1.0 / (probability[indxs] * sample_size))
    timeTaken = time.time() - startTime

    return indxs, S, labels, weights, timeTaken

class RBFNN(CoresetMethod):

    def __init__(self, dst_train, fraction=0.5, random_seed=None, sensitivity_file=None, **kwargs):
        super().__init__(dst_train, fraction, random_seed)
        self.sensitivity_file = sensitivity_file

    def select(self, **kwargs):
        n = len(self.dst_train)
        sample_size = int(n * self.fraction)  # size of the coreset

        # DataLoader with batch_size equal to the total dataset size
        loader = DataLoader(self.dst_train, batch_size=n)
        X_train, y_train = next(iter(loader))['input'], next(iter(loader))['labels']
        X_train= X_train.reshape(X_train.shape[0], -1).numpy()
        # raise ValueError(str(X_train.shape))
        # Obtain sensitivities

        if self.sensitivity_file is None:
            sensitivity = obtainSensitivity(X_train, w=None, approxMVEE=kwargs.get('approxMVEE', True))
        elif os.path.exists(self.sensitivity_file):
            sensitivity = np.load(self.sensitivity_file)
        else:
            sensitivity = obtainSensitivity(X_train, w=None, approxMVEE=kwargs.get('approxMVEE', True))
            np.save(self.sensitivity_file, sensitivity)

        # Generate the coreset
        coreset_indices, _, _, sample_weights, _ = generateCoreset(X_train, y_train, sensitivity, sample_size=sample_size, weights=None, SEED=self.random_seed)

        return {"indices": coreset_indices, "weights": sample_weights}