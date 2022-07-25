''' This file contains helper functions for creating a data set'''

from cgi import test
from tkinter import W
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
import matplotlib.pyplot as plt
import torch

# 320 per cluster (256 training, 64 testing)
def gen_clusters(num_clusters=3,num_samples=960,dim=2,std=25,xy_range=(0,500),test_split=0.2):
    # generate the data
    X,Y = make_blobs(n_samples=num_samples,centers=num_clusters,n_features=dim,center_box=xy_range,cluster_std=std)
    split = int(test_split*num_samples)
    return X[split:],Y[split:],X[:split],Y[:split]
    

def show_clusters(X_train,Y_train,X_test=None,Y_test=None,model=None,wrong_idxs=None):

    fig, ax = plt.subplots()

    # use the trained model to find the decision regions
    if model != None:
        # sample equally spaced points from the entire region 
        from torch.autograd import Variable
        x_span = np.linspace(-100,600,1000)/500
        y_span = np.linspace(-100,600,1000)/500
        xx, yy = np.meshgrid(x_span, y_span)

        # make a prediction on each point
        preds = model(Variable(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()))
        pred_labels = np.argmax(preds.detach().numpy(),axis=1)
        z = np.array(pred_labels).reshape(xx.shape)

        # plot a contour using the coordinates and the prediction
        ax.contourf(xx, yy, z,cmap=plt.cm.brg,alpha=0.1)

    # plot the training cluster points
    plt.scatter(X_train[:,0],X_train[:,1],alpha=0.2,s=3,c=Y_train.reshape(Y_train.shape[0]), cmap=plt.cm.brg)
    plt.title("Decision Regions")

    # plot the testing cluster points, make the misclassified points larger and outlined in white
    # and the corerctly classified points outlined in black
    sizes = np.ones(Y_test.shape)+2
    sizes[wrong_idxs] = 8
    line_widths = sizes-2.5
    line_widths[wrong_idxs] = 0.5
    edgecolors = ['k']*Y_test.shape[0]
    for i in wrong_idxs:
        edgecolors[i] = 'w'
    plt.scatter(X_test[:,0],X_test[:,1],s=sizes,c=Y_test.reshape(Y_test.shape[0]), cmap=plt.cm.brg,edgecolors=edgecolors,linewidths=line_widths)
    plt.show()