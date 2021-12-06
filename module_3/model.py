import random

import numpy as np

from logger import logger


class Model(object):

    def __init__(self):
        pass


class Kmean(Model):

    def __init__(self, k):
        super().__init__()
        self.k = k

    def _validate_data(self, X):
        pass

    def get_random_vector(self, dimension, low=0, high=255):

        a = np.empty(dimension)

        for i in range(dimension):
            a[i] = random.randint(low, high)
        return a

    def dist(self, x, y):
        return np.sqrt(
            np.sum((np.array(x) - np.array(y)) * (np.array(x) - np.array(y)))
        )

    def cluster_image(self, X, num_iter=100):
        self._validate_data(X)
        X = np.array(X)
        n, dimension = X.shape
        self.centriods = []
        for i in range(self.k):
            self.centriods.append(self.get_random_vector(dimension, 0, 255))

        self.centriods = np.array(self.centriods)

        for i in range(num_iter):

            dist_matrix = np.sqrt(
                np.sum(np.square(np.expand_dims(X, 1) - self.centriods),
                       axis=-1))

            cluster_ids = np.argmin(dist_matrix, axis=-1)

            for cluster_id in np.unique(cluster_ids):
                self.centriods[cluster_id] = np.mean(
                    X[np.where(cluster_ids == cluster_id)], axis=0)

            # logger.info("Iteration {} average distance {}".format(
            #     i, np.sum(np.min(dist_matrix, axis=-1)) / n))

        # self.centriods = self.centriods.astype(np.uint8)
        return self.centriods

    def predict_cluster(self, X):
        return np.argmin(
            np.sum(np.square(np.expand_dims(X, 1) - self.centriods),
                   axis=-1), axis=-1)
