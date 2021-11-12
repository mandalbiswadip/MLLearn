import os

import pandas as pd


def raise_no_file_error(file):
    if not os.path.exists(file):
        raise FileNotFoundError("File not found: {}".format(file))


def get_clauses_examples_count(path, file_ext=".csv", file_splitter="_"):
    filename = os.path.basename(path)

    file_splits = filename.split(file_splitter)

    _, clauses = file_splits[1][0], file_splits[1][1:]

    _, examples = file_splits[2][0], file_splits[2][1:]
    examples = examples.replace(file_ext, "")
    return int(clauses), int(examples)


def get_test_validation_file(train_file, file_splitter="_", file_ext=".csv"):
    """
    get validation and test file given the training file
    Args:
        train_file: string
            train file path
        file_splitter:
        file_ext:

    Returns: tuple
        validation file path, test file path

    """

    filename = os.path.basename(train_file)
    dirname = os.path.dirname(train_file)

    file_splits = filename.split(file_splitter)

    c, clauses = file_splits[1][0], file_splits[1][1:]

    d, examples = file_splits[2][0], file_splits[2][1:]
    examples = examples.replace(file_ext, "")

    # should not raise if everything is parsed correctly
    int(clauses)
    int(examples)

    test_file_name = "test_{}{}_{}{}{}".format(
        *[c, clauses, d, examples, file_ext])
    val_file_name = "valid_{}{}_{}{}{}".format(
        *[c, clauses, d, examples, file_ext])

    test_file = os.path.join(dirname, test_file_name)
    val_file = os.path.join(dirname, val_file_name)

    raise_no_file_error(test_file)
    raise_no_file_error(val_file)
    return val_file, test_file


def get_feature_target(train_path):
    """
    Args:
        train_path: str
            train data path
    Returns: tuple of numpy array
        features, target
    """
    data = pd.read_csv(train_path)
    features = data.filter(data.columns[:-1]).values
    target = data[data.columns[-1]].values
    return features, target
