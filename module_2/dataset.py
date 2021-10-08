import numpy as np
import pandas as pd

from logger import logger
from util import load_text


class Dataset(object):

    def __init__(self, name):
        self.name = name


class MovieDataset(Dataset):

    def __init__(self, name="netflix_dataset"):
        super(MovieDataset, self).__init__(name=name)

    def fit_dataset(self, file, delm=","):
        data = MovieDataset.get_user_data(file, delm=delm)
        data = pd.DataFrame(data)

        movie_ids = np.unique(data["MovieID"].values)
        self.movie_index_mapping = dict()
        for index, movie_id in enumerate(movie_ids):
            self.movie_index_mapping[movie_id] = index

        user_ids = np.unique(data["UserID"].values)
        self.user_index_mapping = dict()
        for index, user_id in enumerate(user_ids):
            self.user_index_mapping[user_id] = index

        data = data.filter(["MovieID", "UserID", "Rating"])

        logger.info(
            "Unique user count: {}".format(len(self.user_index_mapping)))
        logger.info(
            "Unique movie user: {}".format(len(self.movie_index_mapping)))
        self.rating_matrix = np.zeros(
            shape=(len(self.user_index_mapping), len(self.movie_index_mapping)))

        for movie_id, user_id, rating in data.values:
            if user_id in self.user_index_mapping and movie_id in self.movie_index_mapping:
                self.rating_matrix[
                    self.user_index_mapping[user_id], self.movie_index_mapping[
                        movie_id]] = rating
            else:
                logger.warn(
                    "Either user id: {} or movie id: {} is not recognized".format(
                        user_id, movie_id))

    def fit_transform(self, file, delm=","):
        self.fit_dataset(file, delm=delm)
        return self.rating_matrix

    def transform(self, file, delm= ","):
        data = MovieDataset.get_user_data(file, delm=delm)
        data = pd.DataFrame(data)

        if not hasattr(self, "movie_index_mapping"):
            raise ValueError("Dataset need to be fitted using fit_dataset()")
        if not hasattr(self, "user_index_mapping"):
            raise ValueError("Dataset need to be fitted using fit_dataset()")

        data = data.filter(["MovieID", "UserID", "Rating"])

        rating_matrix = np.zeros(
            shape=(len(self.user_index_mapping), len(self.movie_index_mapping)))

        for movie_id, user_id, rating in data.values:
            if user_id in self.user_index_mapping and movie_id in self.movie_index_mapping:
                rating_matrix[
                    self.user_index_mapping[user_id], self.movie_index_mapping[
                        movie_id]] = rating
            else:
                logger.warn(
                    "Either user id: {} or movie id: {} is not recognized".format(
                        user_id, movie_id))

        return rating_matrix

    def get_user_movie_index(self, file, delm=","):
        data = MovieDataset.get_user_data(file, delm=delm)
        data = pd.DataFrame(data)
        if not hasattr(self, "movie_index_mapping"):
            raise ValueError("Dataset need to be fitted using fit_dataset()")
        if not hasattr(self, "user_index_mapping"):
            raise ValueError("Dataset need to be fitted using fit_dataset()")


        data.MovieID = data.MovieID.apply(lambda x:self.movie_index_mapping[x])
        data.UserID = data.UserID.apply(lambda x:self.user_index_mapping[x])
        return data

    @staticmethod
    def get_user_data(file, delm=","):
        text = load_text(file)
        result = []
        for line in text.splitlines():
            if len(line.split(delm)) > 2:
                result.append({
                    "MovieID": line.split(delm)[0],
                    "UserID": line.split(delm)[1],
                    "Rating": line.split(delm)[2]}
                )
            else:
                result.append({
                    "MovieID": line.split(delm)[0],
                    "UserID": line.split(delm)[1]}
                )

        return result
