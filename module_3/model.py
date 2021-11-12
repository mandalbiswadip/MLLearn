from copy import deepcopy
import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm

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
            d = defaultdict(list)

            distant_sum = 0
            for point in X:

                min_dist, min_distance_index = float("inf"), 0 
                for index, centroid in enumerate(self.centriods):
                    dist = self.dist(centroid, point)
                    if dist < min_dist:
                        min_dist = dist
                        min_distance_index = index
                distant_sum += min_dist
                d[min_distance_index].append(point)
            logger.info("Iteration {} average distance {}".format(i, distant_sum/n))

            for centroid_id, data_list in d.items():
                self.centriods[centroid_id] = np.mean(np.array(data_list), axis=0)
        return self.centriods

    def predict_cluster(self, X):
        cluster_ids = []
        for point in X:

            min_dist, min_distance_index = float("inf"), 0
            for index, centroid in enumerate(self.centriods):
                dist = self.dist(centroid, point)
                if dist < min_dist:
                    min_dist = dist
                    min_distance_index = index
            cluster_ids.append(min_distance_index)
        return cluster_ids





        
        
