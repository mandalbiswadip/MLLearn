#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from dataset import MovieDataset
from logger import logger
from model import CollaborativeFilteringModel

# Only to avoid memory errors
BATCH_SIZE = 10000

path = "../../hw2/netflix/"

# In[6]:


train_text_path = os.path.join(path, "TrainingRatings.txt")
test_text_path = os.path.join(path, "TestingRatings.txt")


movie = MovieDataset()
train_matrix = movie.fit_transform(train_text_path)

model = CollaborativeFilteringModel()

logger.info("training ...")
model.fit_model(train_matrix)

test_pairs = movie.get_user_movie_index(test_text_path).values

all_predictions = []

i = 0
for _ in tqdm(range(int(len(test_pairs) / BATCH_SIZE) + 1)):
    pred = model.predict(test_pairs[i:i+BATCH_SIZE])
    all_predictions.extend(pred)
    i += BATCH_SIZE

test_rating = np.array([float(x) for x in test_pairs[..., 2]])

logger.info("MAE: {}".format(mean_absolute_error(test_rating, all_predictions)))
logger.info(
    "RMSE: {}".format(mean_squared_error(test_rating, pred, squared=False)))
