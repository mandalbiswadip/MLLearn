from sklearn.datasets import fetch_openml

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

X = X / 255.

x_train, y_train = X[:60000], y[:60000]
x_test, y_test = X[60000:], y[60000:]

from sklearn.neighbors import KNeighborsClassifier

for n_neighbors in [5, 10, 20, 30, 50]:
    for weights in ["uniform", "distance"]:
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights
        )

        model.fit(x_train, y_train)
        print("n_neighbors {}, weights {}".format(
            n_neighbors, weights)
        )
        print(100 * (1 - model.score(x_test, y_test)))