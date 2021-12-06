from __future__ import print_function

import sys
import os
import glob

from CLT_class import CLT
from Util import *

from sklearn.utils import resample


class RF_MIXTURE_CLT():

    def __init__(self):
        self.n_components = 0  # number of components
        self.mixture_probs = None  # mixture probabilities
        self.clt_list = []  # List of Tree Bayesian networks

    def generate_random_data(self, low=0, high=2, size=None):
        """generate random dataset"""
        return np.random.randint(low, high=high, size=size)

    '''
        Learn Mixtures of Trees using the EM algorithm.
    '''

    def learn(self, dataset, n_components=2, r=0):
        self.n_components = n_components
        n = dataset.shape[0]
        # For each component and each data point, we have a weight

        # Uniform mixture probabilities
        self.mixture_probs = np.array([1/n_components for _ in range(n_components)])

        # Randomly initialize the chow-liu trees
        # generating randomd data to randomly initialize probablities of chow-liu trees
        for _ in range(n_components):
            boot_dataset = resample(
                dataset, replace=True,
                n_samples=n,
                random_state=1
            )
            clt = CLT()
            clt.learn(boot_dataset, r)
            self.clt_list.append(clt)

    """
        Compute the log-likelihood score of the dataset
    """

    def computeLL(self, dataset: np.array):
        ll = 0.0
        n = dataset.shape[0]

        for i in range(n):
            data_likelihood = 0.
            for k in range(self.n_components):
                data_likelihood += self.mixture_probs[k] * self.clt_list[
                    k].getProb(dataset[i])
            ll += np.log(data_likelihood)
        return ll


if __name__ == "__main__":
    import pandas as pd

    k_values = [2, 5, 10, 20]

    # r values needs to be less than 10000(or 5100 if you consider unique values)
    # the selected r values are spread across available options to see a trend
    # extreme values such as r = 1 or r = 4000 are avaoided as we already have an intuition on performance for them 

    r_values = [100, 500,  1000, 2000]    

    result_logger = []
    
    # After you implement the functions learn and computeLL,
    # you can learn a mixture of trees using
    # To learn Chow-Liu trees, you can use
    folder_path = sys.argv[1]

    data_path_pattern = os.path.join(folder_path, "*.ts.data")

    for data_path in glob.glob(data_path_pattern):
        print("WORKING ON DATAFILE: ", data_path)
        data_results = []

        dataset = Util.load_dataset(data_path)
        val_dataset = Util.load_dataset(data_path.replace(".ts.data", ".valid.data"))
        test_dataset = Util.load_dataset(data_path.replace(".ts.data", ".test.data"))


        for i in range(len(k_values)):
            for r in r_values:
                print("Trying ncomponents = {} and r = {}\n".format(k_values[i], r))
                result_dict = dict()

                rf_clt = RF_MIXTURE_CLT()
                ncomponents = k_values[i]  # number of components
                result_dict["ncomponents"] = ncomponents
                result_dict["r"] = r

                rf_clt.learn(dataset, ncomponents, r = r)

                # To compute average log likelihood of a dataset w.r.t. the mixture, you can use

                result_dict["validation_log_likelihood"] = rf_clt.computeLL(val_dataset) / val_dataset.shape[0]
                print("Validation Log likelihood: ", result_dict["validation_log_likelihood"])
                data_results.append(result_dict)
        


        selected_setting = sorted(
            data_results, reverse=True, 
            key=lambda x: x["validation_log_likelihood"]
        )[0]


        log_likelihood = []
        for i in range(5):
            rf_clt = RF_MIXTURE_CLT()
            rf_clt.learn(
                dataset, 
                selected_setting["ncomponents"], 
                selected_setting["r"]
            )
            log_likelihood.append(rf_clt.computeLL(test_dataset) / test_dataset.shape[0])
        
        log_likelihood = np.array(log_likelihood)
        print("Log likelihood trials: ", log_likelihood)        

        result_logger.append({
            "data" : data_path,
            "best_n_components": selected_setting["ncomponents"],
            "best_r": selected_setting["r"],
            "log_likelihood_mean": np.mean(log_likelihood),
            "log_likelihood_standard_deviation": np.std(log_likelihood)

        })

        pd.DataFrame(result_logger).to_csv("RF_clt_v1.csv")
        print("="*50)
