1. Extract the zip. Activate the appropriate virtual environment. Run the following to install needed libraries:
```shell
pip install -r requirements.txt
```

2. ***Collaborative Filtering***

```python
 python hw_collaborative_filtering.py data_path
```

`data_path` is the folder where netflix dataset resides with the following files:<br>
`1. TrainingRatings.txt`<br>

`2. TestingRatings.txt`

3. ***MNIST***

Run the following commends to run SVM, MLPClassifier and K-Nearest Neighbour respectively

1. ```python SVM_mnist.py```

2. ```python MLPClassifier_mnist.py```

3. ```python KNN_mnist.py```

Note that this runs all the hyper-parameter tuning as well and may take long time to finish.

Refer to report for all scores obtained.