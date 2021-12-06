from __future__ import print_function

import sys

from CLT_class import CLT
from Util import *
import glob
import os


class MIXTURE_CLT():

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

    def learn(self, dataset, n_components=2, max_iter=50, epsilon=1e-5):
        self.n_components = n_components
        n = dataset.shape[0]
        # For each component and each data point, we have a weight
        weights = np.zeros((n_components, n))

        # Randomly initialize the mixture probabilities
        self.mixture_probs = np.random.dirichlet(np.ones(n_components))

        # Randomly initialize the chow-liu trees
        # generating randomd data to randomly initialize probablities of chow-liu trees
        for _ in range(n_components):
            rand_dataset = self.generate_random_data(low=0, high=2,
                                                     size=(dataset.shape))
            clt = CLT()
            clt.learn(rand_dataset)
            self.clt_list.append(clt)

        prev_ll, ll = 0., 0.
        for itr in range(max_iter):
            print("Iteration..{}".format(itr))
            # E-step: Complete the dataset to yield a weighted dataset
            for i in range(n):
                for j in range(n_components):
                    weights[j, i] = self.clt_list[j].getProb(dataset[i]) * \
                                    self.mixture_probs[j]

            # normalize across the n_component axis.
            # for a given datapoint all component weight should sum up-to one
            # We store the weights in an array weights[ncomponents,number of points]
            weights = weights / np.sum(weights, axis=0)

            # M-step: Update the Chow-Liu Trees and the mixture probabilities
            for k in range(n_components):
                tau_k = np.sum(weights[k])
                self.mixture_probs[k] = tau_k / n
                self.clt_list[k] = self.clt_list[k].update(dataset, weights[k])

            prev_ll = ll
            ll = self.computeLL(dataset) / n
            print("LL : {}".format(ll))
            
            if abs(ll - prev_ll) <= epsilon:
                print("Stopping iteration since log likelihood increase is very small.")
                break

            

    """
        Compute the log-likelihood score of the dataset
    """

    def computeLL(self, dataset: np.array):
        ll = 0.0
        n = dataset.shape[0]

        for i in range(n):
            data_likelihood = 0.
            for k in range(self.n_components):
                data_likelihood += self.mixture_probs[k] * self.clt_list[k].getProb(dataset[i])
            ll += np.log(data_likelihood)
        return ll


if __name__ == "__main__":
    import pandas as pd

    k_values = [2, 5, 10, 20]

    result_logger = []


    # folder_path = "/Users/biswadipmandal/Documents/MSCS/Fall_21/CS_6375_ML/homeworks/hw4/dataset"
    folder_path = sys.argv[1]


    # After you implement the functions learn and computeLL,
    # you can learn a mixture of trees using
    # To learn Chow-Liu trees, you can use
    data_path_pattern = os.path.join(folder_path, "*.ts.data")

    for data_path in glob.glob(data_path_pattern)[9:10]:
        print("WORKING ON DATAFILE: ", data_path)
        data_results = []

        dataset = Util.load_dataset(data_path)
        val_dataset = Util.load_dataset(data_path.replace(".ts.data", ".valid.data"))
        test_dataset = Util.load_dataset(data_path.replace(".ts.data", ".test.data"))


        for i in range(len(k_values)):
            print("Trying ncomponents = ", k_values[i])
            result_dict = dict()

            mix_clt = MIXTURE_CLT()
            ncomponents = k_values[i]  # number of components
            max_iter = 50  # max number of iterations for EM
            epsilon = 1e-1  # converge if the difference in the log-likelihods between two iterations is smaller 1e-1
            result_dict["ncomponents"] = ncomponents
            result_dict["max_iter"] = max_iter
            result_dict["epsilon"] = epsilon

            mix_clt.learn(dataset, ncomponents, max_iter, epsilon)

            # To compute average log likelihood of a dataset w.r.t. the mixture, you can use

            result_dict["validation_log_likelihood"] = mix_clt.computeLL(val_dataset) / val_dataset.shape[0]
            data_results.append(result_dict)
        


        selected_setting = sorted(
            data_results, reverse=True, 
            key=lambda x: x["validation_log_likelihood"]
        )[0]


        log_likelihood = []
        for i in range(5):
            mix_clt = MIXTURE_CLT()
            mix_clt.learn(
                dataset, 
                selected_setting["ncomponents"], 
                selected_setting["max_iter"], 
                selected_setting["epsilon"]
            )
            log_likelihood.append(mix_clt.computeLL(test_dataset) / test_dataset.shape[0])
        
        log_likelihood = np.array(log_likelihood)
        print("Log likelihood trials: ", log_likelihood)        

        result_logger.append({
            "data" : data_path,
            "best_n_components": selected_setting["ncomponents"],
            "log_likelihood_mean": np.mean(log_likelihood),
            "log_likelihood_standard_deviation": np.std(log_likelihood)

        })

        pd.DataFrame(result_logger).to_csv("Mixture_clt_v9_10.csv")
        print("="*50)