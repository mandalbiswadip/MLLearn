decision_tree_parameter_choices = {
    'criterion': ('gini', 'entropy'),
    'max_depth': [3, 5, 10, None],
    # maximum is log2(num_examples)
    'splitter': ('best', 'random'),
    'max_features': (
        'auto', 'sqrt', 'log2', None)
}

bagging_classifier_parameter_choices = {
    'n_estimators': (5, 10, 20, 25),
    # increasing more will take very long to train

    'max_samples': (.6, .8, 1),
    'bootstrap': (True, False),
    'max_features': (.6, .8, 1),
    'bootstrap_features': (True, False)

}


random_forest_parameter_choices = {
    'n_estimators': (50, 100, 200),
    'criterion': ('gini', 'entropy'),
    'max_depth': [3, 5, 10, None],
    'max_features': ('auto', 'sqrt', 'log2')
}

gradient_boosting_parameter_choices = {
    'loss' : ('deviance', 'exponential'),
    'learning_rate' : (0.1, .001, .0001),
    'n_estimators' : (50, 100, 200),
    'max_features': ('auto', 'sqrt', 'log2')
}
