import math
from typing import List


def validate_input(documents: List, labels: List):
    if len(documents) == 0:
        raise ValueError("Empty input for features!!")
    if len(labels) == 0:
        raise ValueError("Empty input for labels!!")

    assert len(documents) == len(
        labels), "count of documents is {} while are count of labels is {}".format(
        len(documents), len(labels))


# Smart implementation of sigmoid to avoid overflow error
def sigmoid(z):
    if z < 0:
        return 1 - 1 / (1 + math.exp(z))
    else:
        return 1 / (1 + math.exp(-z))
