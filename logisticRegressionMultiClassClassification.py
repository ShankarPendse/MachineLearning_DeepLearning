import scipy.io as scipyio
import numpy as np
import scipy.optimize as opt
import logisticRegression as lr
import time

num_labels = 10
lambda_value = 0.1
  
def oneVsAll(X,y,num_labels,lambda_value):
    m = X.shape[0]
    n = X.shape[1]
    all_theta = np.zeros((num_labels,n+1))
    X = np.c_[np.ones(X.shape[0]),X]
    initial_theta = np.zeros(n+1).reshape(n+1,1)
    for i in range(1,num_labels+1):
        all_theta[i-1,:] = np.array([opt.fmin_cg(lr.regularized_costFunction,fprime = lr.regularized_gradient,x0 = initial_theta,args=(X,y==i,lambda_value))])
    return all_theta


def predictOneVsAll(all_theta,X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    predictions = np.zeros((X.shape[0],1))
    X = np.c_[np.ones(X.shape[0]),X]
    hypothesis = lr.sigmoid(X@all_theta.transpose())
    predictions = np.argmax(hypothesis,axis=1)+1
    return predictions

def calculateAccuracy(predictions,y):
    no_of_predictions = 0
    m = len(predictions)

    for i in range(m):
        if(predictions[i] == y[i]):
            no_of_predictions += 1
            
    return no_of_predictions*(100/m)


if(__name__ == '__main__'):
    start_time = time.time()
    data = scipyio.loadmat('ex3data1.mat')
    X = data['X']
    y = data['y']

    y = y.reshape(y.shape[0],1)
    all_theta = oneVsAll(X,y,num_labels,lambda_value)

    predictions = predictOneVsAll(all_theta,X)
    accuracy = calculateAccuracy(predictions,y)

    print("predictions size: ",predictions.shape)
    print("y size: ",y.shape)
    print("Accuracy is : ",accuracy,"%")
    print("\nIt took ",time.time()-start_time," seconds to complete the execution\n")
