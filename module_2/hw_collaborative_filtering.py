#!/usr/bin/env python
# coding: utf-8

import os
from time import time
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dataset import MovieDataset
from logger import logger
from model import CollaborativeFilteringModel

start = time()

path = sys.argv[1]

train_text_path = os.path.join(path, "TrainingRatings.txt")
test_text_path = os.path.join(path, "TestingRatings.txt")

movie = MovieDataset()
train_matrix = movie.fit_transform(train_text_path)

model = CollaborativeFilteringModel()

logger.info("training ...")
model.fit_model(train_matrix)

test_pairs = movie.get_user_movie_index(test_text_path).values

pred = model.predict(test_pairs)
test_rating = np.array([float(x) for x in test_pairs[..., 2]])

logger.info("\nMAE: {}".format(mean_absolute_error(test_rating, pred)))
logger.info(
    "RMSE: {}".format(mean_squared_error(test_rating, pred, squared=False)))

delta_t = time() - start
logger.info(
    "\nTotal time taken: {} mins {} secons".format(int(delta_t / 60), int(delta_t % 60)))
