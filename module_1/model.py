import math
from collections import defaultdict
from time import time
from typing import List

import numpy as np

from logger import logger
from util import validate_input, sigmoid


class Model(object):
    def __init__(self):
        pass

    def train(self):
        return NotImplemented


class MultinomialNaiveBayesModel(Model):
    """
    parsing input text
    """

    def __init__(self, laplace_smoothing=True):
        super(MultinomialNaiveBayesModel, self).__init__()
        self.class_counter = {}
        self.class_prior = {}
        self.cond_prob_estimates = defaultdict(dict)
        self.laplace_smoothing = laplace_smoothing

    def _validate_input(self, documents: List, labels: List):
        validate_input(documents, labels)

    def train(self, features: List, labels: List):
        """
        :param features:
        :param labels:
        :return:
        """
        self._validate_input(features, labels)
        self.num_class = len(set(labels))
        labels = labels

        for label in labels:
            self.class_counter[label] = self.class_counter.get(label, 0) + 1

        for k, v in self.class_counter.items():
            self.class_prior[k] = math.log(v / len(labels))

        class_feature_counter = defaultdict(dict)
        for row, label in zip(features, labels):

            # iterate through features in a single data point
            for i in range(len(row)):
                class_feature_counter[label][i] = class_feature_counter[
                                                      label].get(i, 0) + row[i]

        for label in self.class_prior:
            d = class_feature_counter[label]
            den = sum(d.values())
            if self.laplace_smoothing:
                den += len(features[0])

            for i in range(len(features[0])):
                if self.laplace_smoothing:
                    num = 1
                if i in d:
                    num += d[i]
                self.cond_prob_estimates[label][i] = math.log(num / den)

    def predict(self, features: List):
        predictions = []
        for row in features:
            class_scores = {}
            for label, log_prior in self.class_prior.items():
                class_scores[label] = log_prior

            for i in range(len(row)):
                for label, _ in self.class_prior.items():
                    class_scores[label] += row[i] * self.cond_prob_estimates[
                        label].get(
                        i, 0)
            predictions.append(
                sorted(class_scores.items(), reverse=True, key=lambda x: x[1])[
                    0][0])
        return predictions


class DiscreteNaiveBayesModel(Model):
    """
    Discrete Naive Bayes Model
    """

    def __init__(self, laplace_smoothing=True):
        super(DiscreteNaiveBayesModel, self).__init__()
        self.class_counter = {}
        self.class_prior = {}
        self.cond_prob_estimates = defaultdict(dict)
        self.laplace_smoothing = laplace_smoothing

    def _validate_input(self, documents: List, labels: List):
        validate_input(documents, labels)

    def train(self, features: List, labels: List):
        """
        :param features:
        :param labels:
        :return:
        """
        self._validate_input(features, labels)
        self.num_class = len(set(labels))

        for label in labels:
            self.class_counter[label] = self.class_counter.get(label, 0) + 1

        for k, v in self.class_counter.items():
            self.class_prior[k] = math.log(v / len(labels))

        class_feature_counter = defaultdict(dict)
        for row, label in zip(features, labels):

            # iterate through features in a single data point
            for i in range(len(row)):
                if row[i] > 0:
                    class_feature_counter[label][i] = class_feature_counter[
                                                          label].get(i, 0) + 1

        for label in self.class_prior:
            d = class_feature_counter[label]
            # den = self.class_counter[label]
            den = sum(d.values())
            if self.laplace_smoothing:
                den += len(features[0])
                # den += 1
            num = 0
            for i in range(len(features[0])):
                if self.laplace_smoothing:
                    num = 1
                if i in d:
                    num += d[i]
                self.cond_prob_estimates[label][i] = math.log(num / den)

    def predict(self, features: List):
        predictions = []
        for row in features:
            class_scores = {}
            for label, log_prior in self.class_prior.items():
                class_scores[label] = log_prior

            for i in range(len(row)):
                for label, _ in self.class_prior.items():
                    if row[i] > 0:
                        class_scores[label] += self.cond_prob_estimates[
                            label].get(
                            i, 0)
            predictions.append(
                sorted(class_scores.items(), reverse=True, key=lambda x: x[1])[
                    0][0])
        return predictions


class LogisticRegression(Model):

    def __init__(self, regularizer: str = "l2", reg_costant=0.1,
                 learning_rate=0.01,
                 max_iterations=100, n_features=None, verbose=True):
        self.regularizer = regularizer
        self.reg_costant = reg_costant
        self.lr = learning_rate
        self.max_iterations = max_iterations
        self.n_features = n_features
        self.verbose = verbose
        self.sigmoid = np.vectorize(sigmoid)

    def _validate_input(self, features: List, labels: List):
        validate_input(documents=features, labels=labels)

    def _append_bias_dimension(self, features):
        # TODO improve this
        new_features = []
        for row in features:
            row = list(row)
            row.append(1)  # appending new dimension for bias handling
            new_features.append(row)
        return new_features

    def train(self, features=None, labels=None):
        if self.n_features is None:
            self.n_features = features[0].__len__()
        else:
            if features[0].__len__() != self.n_features:
                raise ValueError(
                    "Feature vector length is {} \n needed length is {}!!".format(
                        features[0].__len__(), self.n_features))
        self._validate_input(features, labels)

        self.unique_labels = set(labels)
        if len(self.unique_labels) > 2:
            raise ValueError("Implements only supports binary classification!!")

        self.label_map = {}

        self.weights = np.random.normal(0, 0.1, self.n_features + 1)

        features = self._append_bias_dimension(features)

        for index, l in enumerate(self.unique_labels):
            self.label_map[l] = index
        logger.info("Label mapping {}\n".format(self.label_map))

        labels = [self.label_map[label] for label in labels]  # convert to 0/1

        if not isinstance(features, np.ndarray):
            features = np.array(features)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        # Gradient descent
        for epoch in range(self.max_iterations):
            if self.verbose and epoch % 10 == 0:
                logger.info("At training step {}".format(epoch))

            # yl - P(yl=1|xl, weight)
            a = features @ self.weights  # matrix multiply
            prob_error = labels - self.sigmoid(a)
            if self.verbose and epoch % 10 == 0:
                logger.info("Prob error estimate: {}\n".format(sum(prob_error)))

            for i, wi in enumerate(self.weights):

                # Do not regularize the bias. No point in doing that
                if i == len(self.weights) - 1:
                    wi = wi + self.lr * np.sum(features[..., i] * prob_error)
                else:
                    wi = wi + self.lr * np.sum(features[
                                                   ..., i] * prob_error) - self.lr * self.reg_costant * wi
                self.weights[i] = wi

    def decision(self, x, reverse_label_map):
        if x > 0:
            return reverse_label_map[1]
        else:
            return reverse_label_map[0]

    def predict(self, features):
        """
        Inference
        :param features: feature vectors as array or list
        :return: list of predictions
        """
        features = self._append_bias_dimension(features)

        if not hasattr(self, "weights"):
            raise ValueError("Model hasn't been trained yet!!")

        if not hasattr(self, "reverse_label_map"):
            self.reverse_label_map = {v: k
                                      for k, v in self.label_map.items()}
            self.decide = np.vectorize(
                lambda x: self.decision(x, self.reverse_label_map))

        if not isinstance(features, np.ndarray):
            features = np.array(features)

        s = features @ self.weights
        return self.decide(s)
