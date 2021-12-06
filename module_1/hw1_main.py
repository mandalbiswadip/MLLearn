import argparse
import os

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from dataset import TextReader, TextData
from logger import logger
from model import LogisticRegression, MultinomialNaiveBayesModel, \
    DiscreteNaiveBayesModel


def get_data(folder, feature_type="bow"):
    """
    folder: with train and test folders
    :param folder:
    :param feature_type: "bow" or "bernoulli"
    :return: train_features, train_labels, test_features, test_labels,
    """
    train_folder = os.path.join(folder, "train")
    test_folder = os.path.join(folder, "test")
    reader = TextReader.read_from_folder(
        train_folder, unique_labels=["ham", "spam"]
    )
    docs, labels = reader.get_data()
    text_data = TextData(docs)
    if feature_type == "bow":
        features = text_data.transform_bag_of_words(docs)
    elif feature_type == "bernoulli":
        features = text_data.transform_bernoulli(docs)

    test_reader = TextReader.read_from_folder(
        test_folder, unique_labels=["ham", "spam"]
    )
    test_docs, test_labels = test_reader.get_data()

    if feature_type == "bow":
        test_features = text_data.transform_bag_of_words(test_docs)
    elif feature_type == "bernoulli":
        test_features = text_data.transform_bernoulli(test_docs)

    return features, labels, test_features, test_labels


def tune_sgdclassifier(features, labels, test_features, test_labels):
    parameters = {'loss': (
        'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),
        'penalty': ('l2', 'l1', 'elasticnet'),
        "alpha": (1e-2, 1e-3, 1e-4, 1e-5),  # regularize term
        "learning_rate": (
            "constant", "optimal", "invscaling", "adaptive"),
        # lr decay schedule
        "eta0": (1e-3, 1e-4)
    }
    md = SGDClassifier()
    clf = GridSearchCV(md, parameters)
    clf.fit(features, labels)
    pred = clf.predict(test_features)
    logger.info("Optimum SGD Classifier {}".format(clf.best_params_))
    print_accuracy(test_labels, pred)
    return clf


def optimum_regularize(train_features, train_labels, dev_features, dev_labels,
                       test_features, test_labels):
    final_model, final_lambda = None, None
    max_accuracy = 0

    for lamb in [10, 1, .1, .001, .0001, 0.0001, .00001]:
        model = LogisticRegression(reg_costant=lamb, max_iterations=300,
                                   learning_rate=0.0001)
        model.train(features=train_features, labels=train_labels)
        pred = model.predict(dev_features)
        accc = accuracy_score(pred, dev_labels)

        if accc > max_accuracy:
            max_accuracy = accc
            final_model = model
            final_lambda = lamb
    pred = final_model.predict(test_features)
    logger.info("Optimum regularization constant {}".format(final_lambda))
    print_accuracy(test_labels, pred)


def print_accuracy(test_labels, predictions):
    logger.info("Accuracy {}".format(accuracy_score(test_labels, predictions)))
    # logger.info("F1 score {}".format(f1_score(test_labels, predictions, pos_label=None)))
    logger.info("classification report \n {}".format(
        classification_report(test_labels, predictions)))


if __name__ == "__main__":

    # This is the data path used while running in my machine. please provide your data path
    folder = "/Users/biswadipmandal/Documents/MSCS/Fall_21/CS_6375_ML/homeworks/hw1"
    argparser = argparse.ArgumentParser(
        description="Train a spam classification model")
    argparser.add_argument('--data_path', type=str, default=folder,
                           help="folder containing data")
    args = argparser.parse_args()
    folder = args.data_path
    data = {}
    features, labels, test_features, test_labels = get_data(folder,
                                                            feature_type="bow")
    data["bow"] = (features, labels, test_features, test_labels)
    features, labels, test_features, test_labels = get_data(folder,
                                                            feature_type="bernoulli")
    data["bernoulli"] = features, labels, test_features, test_labels

    features, labels, test_features, test_labels = data["bow"]
    # # part 2
    logger.info("Running Multinomial Naive Bayes using bag of words features..")
    model = MultinomialNaiveBayesModel()
    model.train(features, labels)
    pred = model.predict(test_features)
    print_accuracy(test_labels, pred)

    logger.info("\n"*5)

    features, labels, test_features, test_labels = data["bernoulli"]
    # part 3
    logger.info("Running Discrete Naive Bayes using bernoulli features..")
    model = DiscreteNaiveBayesModel()
    model.train(features, labels)
    pred = model.predict(test_features)
    print_accuracy(test_labels, pred)

    logger.info("\n"*5)

    # part 4 - LR
    features, labels, test_features, test_labels = data["bow"]
    logger.info(
        "Tuning Logistic Regression using bag of words to get the best lambda..")
    train_features, dev_features, train_labels, dev_labels = train_test_split(
        features, labels,
        test_size=.3, shuffle=True)
    optimum_regularize(train_features, train_labels, dev_features, dev_labels,
                       test_features, test_labels)

    features, labels, test_features, test_labels = data["bernoulli"]
    logger.info(
        "Tuning Logistic Regression using bernoulli to get the best lambda..")
    train_features, dev_features, train_labels, dev_labels = train_test_split(
        features, labels,
        test_size=.3, shuffle=True)
    optimum_regularize(train_features, train_labels, dev_features, dev_labels,
                       test_features, test_labels)
    logger.info("\n"*5)
    #
    # part 5
    logger.info(
        "Tuning SGD Classifier using bag of words to get the best lambda..")
    features, labels, test_features, test_labels = data["bow"]
    tune_sgdclassifier(features, labels, test_features, test_labels)
    logger.info(
        "Tuning SGD Classifier using bernoulli to get the best lambda..")

    features, labels, test_features, test_labels = data["bernoulli"]
    tune_sgdclassifier(features, labels, test_features, test_labels)
