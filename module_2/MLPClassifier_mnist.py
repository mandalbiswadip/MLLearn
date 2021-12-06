from sklearn.datasets import fetch_openml

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

X = X / 255.

x_train, y_train = X[:60000], y[:60000]
x_test, y_test = X[60000:], y[60000:]

from sklearn.neural_network import MLPClassifier

for learning_rate_init in [0.01, 0.001, 0.0001]:
    for hidden_layer_sizes in [(100), (200, 100), (10, 10)]:
        for activation in ["relu", "tanh", "sigmoid"]:
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                learning_rate_init=learning_rate_init
            )
            model.fit(x_train, y_train)
            print(
                "hidden_layer_sizes {}, activation {}, learning_rate_init {}".format(
                    hidden_layer_sizes, activation, learning_rate_init)
            )
            print(100*(1 - model.score(x_test, y_test)))