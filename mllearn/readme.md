To test the code:
1. Extract the zipfile
2. create a virtual environment using conda and run the following command to install sklearn
   
`conda install -c conda-forge scikit-learn`
   
please note that python3.9 doesn't have a stable scikit-learn release for pip yet, so you must use conda
3. Go inside `hw1` directory
4. run the following command for the assignment:

`python hw1_main.py --data_path folder` 

where `folder` is the path for the **dataset which should contain `train` and `test` folders inside.**

The code runs the following operations' step by step:

1. Multinomial Naive Bayes using bag of words
2. Discrete Naive Bayes using Bernoulli representation
3. My Implementation of logistic regression and finding the optimum lambda for bag of words model
4. My Implementation of logistic regression and finding the optimum lambda for bernoulli model
5. SGDClassifier and GridSearchCV to tune hyper-parameters for bag of words model -- takes long to run as tuning a lot of parameters
6. SGDClassifier and GridSearchCV to tune hyper-parameters for bernoulli model -- takes long to run as tuning a lot of parameters


Please refer to the report for accuracies.

