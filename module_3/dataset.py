from util import (get_test_validation_file, get_feature_target,
                  get_clauses_examples_count)


class CNFDataset(object):
    """
    CNF dataset generator class
    """

    def __init__(self, train_path):
        self.train_path = train_path
        self.validation_path, self.test_path = self._get_validation_test_set()

    def _get_validation_test_set(self):
        return get_test_validation_file(self.train_path)

    def get_train_data(self):
        return get_feature_target(self.train_path)

    def get_validation_data(self):
        return get_feature_target(self.validation_path)

    def get_test_data(self):
        return get_feature_target(self.test_path)

    def __str__(self):
        clauses, examples = get_clauses_examples_count(self.train_path)
        return "Dataset with {} clauses and {} train examples".format(clauses,
                                                                      examples)

    def __repr__(self):
        clauses, examples = get_clauses_examples_count(self.train_path)
        return "c{}_d{}".format(clauses, examples)
