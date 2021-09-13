import math
from collections import defaultdict
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

    def __init__(self, tokenizer=None, lower_text=True, laplace_smoothing=True):
        super(MultinomialNaiveBayesModel, self).__init__()
        self.vocabulary = {}
        self.class_counter = {}
        self.class_prior = {}
        self.cond_prob_estimates = defaultdict(dict)
        self.tokenizer = tokenizer
        self.lower_text = lower_text
        self.laplace_smoothing = laplace_smoothing

    def tokenize(self, sent):
        if sent is None:
            raise ValueError(
                "Invalid sentence given {}. requires string type".format(sent))
        if self.lower_text:
            sent = sent.lower()
        if self.tokenizer is not None:
            return self.tokenizer.tokenize(sent)
        else:
            return sent.split()

    def _validate_input(self, documents: List, labels: List):
        validate_input(documents, labels)

    def train(self, documents: List, labels: List):

        if documents is None or labels is None:
            raise ValueError("Provide valid documents and labels!!")
        self._validate_input(documents, labels)

        self.num_class = labels.__len__()
        self.labels = labels

        for label in self.labels:
            if label in self.class_counter:
                self.class_counter[label] += 1
            else:
                self.class_counter[label] = 1

        for k, v in self.class_counter.items():
            self.class_prior[k] = math.log(v / len(documents))

        class_feature_counter = defaultdict(dict)
        for document, label in zip(documents, labels):

            document = self.tokenize(document)
            for word in document:
                class_feature_counter[label][word] = class_feature_counter[
                                                         label].get(word, 0) + 1
                self.vocabulary[word] = self.vocabulary.get(word, 0) + 1

        for label in self.class_prior:
            d = class_feature_counter[label]
            den = sum(d.values())
            if self.laplace_smoothing:
                den += len(self.vocabulary)
            for word in self.vocabulary:
                num = 0
                if self.laplace_smoothing:
                    num = 1
                if word in d:
                    num += d[word]
                self.cond_prob_estimates[label][word] = math.log(num / den)

    def predict(self, documents: List):
        predictions = []
        for document in documents:
            document = self.tokenize(document)
            class_scores = {}
            for label, log_prior in self.class_prior.items():
                class_scores[label] = log_prior

            for word in document:
                for label, _ in self.class_prior.items():
                    class_scores[label] += self.cond_prob_estimates[label].get(
                        word, 0)
            predictions.append(
                sorted(class_scores.items(), reverse=True, key=lambda x: x[1])[
                    0])
        return predictions


class LogisticRegression(Model):

    def __init__(self, regularizer: str = "l2", reg_costant=0.1, learning_rate=0.01,
                 max_iterations=100, n_features=None, verbose=True):
        self.regularizer = regularizer
        self.reg_costant = reg_costant
        self.lr = learning_rate
        self.max_iterations = max_iterations
        self.n_features = n_features
        self.verbose = verbose

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

        # Gradient descent
        for epoch in range(self.max_iterations):
            if self.verbose:
                logger.info("At training step {}".format(epoch))
                logger.info("Sum of weights {}".format(sum(self.weights)))

            # yl - P(yl=1|xl, weight)
            prob_error = []
            for feature, label in zip(features, labels):
                # single row
                z = sum([x * y for x, y in zip(self.weights, feature)])
                prob_error.append(label - sigmoid(z))
            if self.verbose:
                logger.info("Prob error estimate: {}\n".format(sum(prob_error)))

            # TODO can improve this by keeping a transpose copy of features
            for i, wi in enumerate(self.weights):

                # Do not regularize the bias. No point in doing that
                if i == len(self.weights) -1:
                    wi = wi + self.lr * sum([x[i] * prob_error[l] for l, x in
                                             enumerate(
                                                 features)])
                else:
                    wi = wi + self.lr * sum([x[i] * prob_error[l] for l, x in
                                         enumerate(
                                             features)]) - self.lr * self.reg_costant * wi
                self.weights[i] = wi

    def predict(self, features):
        """
        Inference
        :param features: list of list
        :return: list of predictions
        """
        features = self._append_bias_dimension(features)

        if not hasattr(self, "weights"):
            raise ValueError("Model hasn't been trained yet!!")

        if not hasattr(self, "reverse_label_map"):
            self.reverse_label_map = {v: k
                                      for k, v in self.label_map.items()}

        predictions = []
        for feature in features:
            s = sum([x * y for x, y in zip(self.weights, feature)])
            if s > 0:
                predictions.append(self.reverse_label_map[1])
            else:
                predictions.append(self.reverse_label_map[0])
        return predictions
