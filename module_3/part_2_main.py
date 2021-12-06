import os
import sys

import numpy as np
from matplotlib import image, pyplot

from logger import logger
from model import Kmean


def run_clustering(file_name, k):
    img = image.imread(file_name)
    data = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

    model = Kmean(k=k)
    model.cluster_image(data, num_iter=50)

    cluster_ids = model.predict_cluster(data)

    revised_data = np.array(model.centriods)[cluster_ids]

    revised_data = revised_data.reshape((img.shape)).astype(np.uint8)
    compressed_image_path = file_name.replace(".jpg",
                                              "{}_compressed.jpg".format(k))
    pyplot.imsave(compressed_image_path,
                  revised_data)

    orig_size = os.stat(file_name).st_size
    comp_size = os.stat(compressed_image_path).st_size

    return orig_size / comp_size


if __name__ == "__main__":
    file_name = sys.argv[1]
    logger.info("Filename {}\n".format(file_name))

    for k in [2, 5, 10, 15, 20]:
        compression_ratios = []
        for i in range(10):
            logger.info("Running clustering for k = {}\n".format(k))
            logger.info("Trial number {}".format(i + 1))
            compression_ratios.append(run_clustering(file_name, k))
        compression_ratios = np.array(compression_ratios)
        logger.info("Average compression Ratio for k {} is {}".format(k,
                                                                      np.mean(
                                                                          compression_ratios)))
        logger.info(
            "Standard Deviation in compression Ratio for k {} is {}".format(k,
                                                                            np.std(
                                                                                compression_ratios)))
