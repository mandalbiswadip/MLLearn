import sys


from model import Kmean
from matplotlib import image, pyplot
import numpy as np

k = 15
file_name = sys.argv[1]
img = image.imread(file_name)

data = img.reshape(img.shape[0] * img.shape[1], img.shape[2])

model = Kmean(k = k)
model.cluster_image(data, num_iter=16)

cluster_ids = model.predict_cluster(data)

revised_data = np.array(model.centriods)[cluster_ids]

revised_data = revised_data.reshape((img.shape)).astype(int)

pyplot.imsave(file_name.replace(".jpg", "_compressed.jpg"), revised_data)

