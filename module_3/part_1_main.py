import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.tree import DecisionTreeClassifier

from data_util import DataFileGenerator
from dataset import CNFDataset
from config_param_choices import (bagging_classifier_parameter_choices,
                                   decision_tree_parameter_choices,
                                   random_forest_parameter_choices,
                                   gradient_boosting_parameter_choices)


def tune_model(model_interface, params, folder, name=""):
    """
    Args:
        model: mix of sklearn model
        params: dict
            hyperparameter choices
        folder: str
            data folder path

    """
    accuracy_logger = []
    for path in DataFileGenerator(folder):

        acc_log = dict()
        dataset = CNFDataset(path)

        acc_log["dataset"] = dataset
        print(dataset)

        # train and validation data
        features, target = dataset.get_train_data()

        val_features, val_target = dataset.get_validation_data()
        X, y = np.concatenate((features, val_features)), np.concatenate(
            (target, val_target))

        test_indices = [-1 for _ in range(len(features))]
        for _ in range(len(val_features)):
            test_indices.append(1)

        # val_features.shape
        model = model_interface()

        parameters = params
        cv = PredefinedSplit(test_fold=test_indices)
        clf = GridSearchCV(model, cv=cv, param_grid=parameters, verbose=0)

        clf.fit(X, y)

        best_model = clf.best_estimator_
        best_params = best_model.get_params()
        acc_log.update(best_params)

        print("Best parameters \n", best_params)

        final_model = model_interface(**best_params)
        final_model.fit(X, y)

        test_features, test_target = dataset.get_test_data()

        score = final_model.score(test_features, test_target)
        acc_log["accuracy"] = score
        print("Accuracy on test set: {}".format(
            score))

        f1 = f1_score(test_target, final_model.predict(test_features))
        acc_log["f1_score"] = f1
        print("F1 score on test set: {}".format(
            f1)
        )
        accuracy_logger.append(acc_log)
        print("-x" * 80)
    print("\nSaving accuracy for {} in {}\n".format(name, name + ".csv"))
    pd.DataFrame(accuracy_logger).to_csv(name + ".csv", index=False)


if __name__ == "__main__":
    folder = sys.argv[1]

    # decision tree classifier
    model = DecisionTreeClassifier
    params = decision_tree_parameter_choices
    print("DecisionTreeClassifier")
    tune_model(
        model_interface=model,
        params=params,
        folder=folder,
        name="DecisionTreeClassifier"
    )

    # bagging

    model = BaggingClassifier
    params = bagging_classifier_parameter_choices
    print("BaggingClassifier")
    tune_model(
        model_interface=model,
        params=params,
        folder=folder,
        name="BaggingClassifier"
    )

    # random forest
    model = RandomForestClassifier
    params = random_forest_parameter_choices
    print("RandomForestClassifier")
    tune_model(
        model_interface=model,
        params=params,
        folder=folder,
        name="RandomForestClassifier"
    )

    # gradient boosting
    # model = GradientBoostingClassifier
    # params = gradient_boosting_parameter_choices
    # print("GradientBoostingClassifier")
    # tune_model(
    #     model_interface=model,
    #     params=params,
    #     folder=folder,
    #     name="GradientBoostingClassifier"
    # )
