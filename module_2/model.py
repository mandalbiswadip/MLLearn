from copy import deepcopy

import numpy as np
from tqdm import tqdm

class Model(object):

    def __init__(self):
        pass


class CollaborativeFilteringModel(Model):

    def __init__(self):
        pass

    def _validate_rating_matrix(self, rating_matrix):
        pass

    def fit_model(self, rating_matrix):
        """
        Args:
            rating_matrix: 2D numpy array, list of list or list of tuples
                shape--> (# user, #count movie)
        :return:

        step:
        1. get and store the W matrix for W(a,i) - shape --> (# user, #user)
        2. get and store average rating of each user --> use numpy.sum and numpy.count_nonzero --> dash(vi)
        3. get p(a,j) for any new pair of user a and movie j
        Returns:

        """
        self._validate_rating_matrix(rating_matrix)
        self.rating_matrix = np.array(rating_matrix)

        #  average rating of each user --> averaged only based on non-zero ratings available
        self.average_ratings = np.sum(self.rating_matrix,
                                      axis=-1) / np.count_nonzero(
            self.rating_matrix, axis=-1)

        # TODO check if deepcopy can be avaoided
        temp_rating_matrix = deepcopy(self.rating_matrix)
        del self.rating_matrix

        temp_rating_matrix[np.nonzero(temp_rating_matrix)] = temp_rating_matrix[
                                                                 np.nonzero(
                                                                     temp_rating_matrix)] \
                                                             - \
                                                             self.average_ratings[
                                                                 np.nonzero(
                                                                     temp_rating_matrix)[
                                                                     0]]

        # TODO check if deepcopy can be avaoided
        self.normalized_ratings = deepcopy(temp_rating_matrix)

        #  divide row by square root of sum of square of row
        b = np.sqrt(
            np.sum(np.square(temp_rating_matrix), axis=-1))[:, None]
        temp_rating_matrix = np.divide(temp_rating_matrix, b,
                                       out=np.zeros_like(temp_rating_matrix),
                                       where=b != 0)
        # temp_rating_matrix = temp_rating_matrix / b
        del b

        temp_rating_matrix = np.nan_to_num(temp_rating_matrix)
        # the w matrix
        self.similarity = temp_rating_matrix @ temp_rating_matrix.transpose()
        self.kappa = np.abs(self.similarity).sum(-1)
        del temp_rating_matrix

    def predict(self, query):
        """
        Args:
            query: list of list or 2d-array of indexes
                (user, movie) pairs of indexes
        Returns:
            1d-array of floats
            predicted ratings
        """
        user_indexes = query[..., 1].astype("int")
        movie_indexes = query[..., 0].astype("int")

        # average_user_rating = self.average_ratings[user_indexes]
        # sim = np.nan_to_num(self.similarity[user_indexes])

        all_prediction = []
        for user_index, movie_index in tqdm(zip(user_indexes, movie_indexes)):
            sim = self.similarity[user_index]
            kappa = 1 / self.kappa[user_index]
            movie_vector = self.normalized_ratings[..., movie_index].transpose()
            avg = self.average_ratings[user_index]
            all_prediction.append(
                avg + np.nan_to_num(kappa *
                                    sim * movie_vector
                                    ).sum()
            )

        return all_prediction

        #
        # return average_user_rating + np.nan_to_num((sim * self.normalized_ratings[
        #     ..., movie_indexes].transpose()).sum(1) / np.abs(sim).sum(-1))

    def get_average_rating(self, query):
        """
        Args:
            query: list of list or 2d-array of indexes
                (user, movie) pairs of indexes
        Returns:
            1d-array of floats
            predicted ratings
        """
        user_indexes = query[..., 1].astype("int")

        average_user_rating = self.average_ratings[user_indexes]
        return average_user_rating
