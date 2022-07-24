''' This file contains helper functions for creating a data set'''

from cgi import test
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
import matplotlib.pyplot as plt

# 320 per cluster (256 training, 64 testing)
def gen_clusters(num_clusters=3,num_samples=960,dim=2,std=25,xy_range=(0,500),test_split=0.2):
    # generate the data
    X,Y = make_blobs(n_samples=num_samples,centers=num_clusters,n_features=dim,center_box=xy_range,cluster_std=std)
    split = int(test_split*num_samples)
    return X[split:],Y[split:],X[:split],Y[:split]
    

def show_clusters(X,Y):
    num_clusters = np.unique(Y)
    for cluster in num_clusters:
        samples = X[Y == cluster]
        plt.scatter(samples[:,0],samples[:,1],label=cluster)
    plt.legend()
    plt.show()