''' This file contains helper functions for creating a data set'''

from cgi import test
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
    

def show_clusters(X_train,Y_train,X_test=None,Y_test=None,model=None,x_wrong=None,y_wrong=None):

    fig, ax = plt.subplots()

    if model != None:
        from torch.autograd import Variable
        x_span = np.linspace(-100,600,1000)
        y_span = np.linspace(-100,600,1000)
        xx, yy = np.meshgrid(x_span, y_span)
        preds = model(Variable(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()))
        pred_labels = np.argmax(preds.detach().numpy(),axis=1)
        z = np.array(pred_labels).reshape(xx.shape)
        ax.contourf(xx, yy, z,cmap=plt.cm.brg,alpha=0.25)

    plt.scatter(X_train[:,0],X_train[:,1],alpha=0.1,s=3,c=Y_train.reshape(Y_train.shape[0]), cmap=plt.cm.brg)
    #plt.scatter(X_test[:,0],X_test[:,1],s=1,c=Y_test.reshape(Y_test.shape[0]), cmap=plt.cm.brg)
    # plt.scatter(x_wrong[:,0],x_wrong[:,1],s=6,c=y_wrong.reshape(y_wrong.shape[0]), cmap=plt.cm.summer)
        # if (X_test != None).all() and (Y_test != None).all():
        #     samples = X_test[Y_test == cluster]
        #     ax.scatter(samples[:,0],samples[:,1],label=cluster,alpha=1,s=1)
            
    plt.show()