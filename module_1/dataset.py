import os
import pathlib
from typing import List


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return content


class Dataset(object):

    def __init__(self):
        pass


class TextReader(object):
    """
    Read Text for training from different formats:
    """

    def __init__(self, documents: List, labels: List):
        self.documents = documents
        self.labels = labels

    @classmethod
    def read_from_folder(cls, home_folder: str,
                         unique_labels: List):
        documents = []
        labels = []
        if os.path.exists(home_folder):
            for label in unique_labels:
                for file in pathlib.Path(os.path.join(home_folder, label)).glob(
                        "*.txt"):
                    labels.append(label)
                    documents.append(read_text(file))
        return cls(documents=documents, labels=labels)

    def get_training_data(self):
        """
        get documents, labels
        :return:
        """
        return self.documents, self.labels

    def get_data(self):
        """same as get_training_data"""
        return self.get_training_data()


class TextData(object):
    """
    Text processing class.
    Inspired from sklearn.feature_extraction.text API
    """
    def __init__(self, documents:List, lower_text=True, tokenizer=None):
        self.vocabulary = {}
        self.word_count = {}
        self.tokenizer = tokenizer
        self.lower_text = lower_text

        self._validate_input(documents)

        count = 0
        for document in documents:
            for word in self.tokenize(document):
                if word not in self.vocabulary:
                    self.vocabulary[word] = count
                    count += 1
                self.word_count[word] = self.word_count.get(word, 0) + 1

    def _validate_input(self, documents:List):
        if documents is None:
            raise ValueError("Provide valid documents!!")

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

    def transform_bag_of_words(self, documents:List):
        """
        convert documents to bag of words (word counts)
        :param documents: list of documents
        :return: list of list
        """
        self._validate_input(documents)
        feature_vector = []
        for document in documents:
            row_vector = [0]*len(self.vocabulary)
            for word in self.tokenize(document):
                if word in self.vocabulary:
                    row_vector[self.vocabulary[word]] += 1
            feature_vector.append(row_vector)
        return feature_vector

    def transform_bernoulli(self, documents):
        """
        convert documents to features based on presence of word (0/1 values)
        :return:
        """
        self._validate_input(documents)
        feature_vector = []
        for document in documents:
            row_vector = [0] * len(self.vocabulary)
            for word in self.tokenize(document):
                if word in self.vocabulary:
                    row_vector[self.vocabulary[word]] = 1
            feature_vector.append(row_vector)
        return feature_vector


