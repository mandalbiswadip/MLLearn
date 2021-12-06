from sklearn.datasets import fetch_openml

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

X = X / 255.

x_train, y_train = X[:60000], y[:60000]
x_test, y_test = X[60000:], y[60000:]

from sklearn.svm import SVC

for c in [0.1, 0.01, 0.001]:
    for kernel in ["linear", "poly", "rbf", "sigmoid", ]:
        for gamma in ["scale", "auto"]:
            model = SVC(kernel=kernel, C=c, gamma=gamma)
            model.fit(x_train, y_train)
            print("c {} gamma {} kernel {}".format(c, gamma, kernel))
            print(100*(1 - model.score(x_test, y_test)))