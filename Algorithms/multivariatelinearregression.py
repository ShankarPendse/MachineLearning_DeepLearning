import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

#set number of iterations and learning rate
iterations=8000
alpha=0.03

def feature_normalize(X):
    X_norm = X
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    for i in range(X_norm.shape[1]-1):
        X_norm[:,i] = (X_norm[:,i]-mu[i])/sigma[i]

    return X_norm,mu,sigma


###Compute the cost (mean squared error)
def computeCost(X,y,theta):
    m = y.shape[0]
    hypothesis = np.matmul(X,theta)
    errors = hypothesis-y
    squared_errors = errors*errors
    sum_squared_errors = np.sum(squared_errors)
    J = sum_squared_errors/(2*m)
    return J


###Gradient descent algorithm
def gradientDescent(X,y,theta,alpha,no_of_iterations):
    m = y.shape[0]
    J_history = np.matrix(np.zeros(no_of_iterations)).transpose()
    no_of_features = X.shape[1]
    featurevalue = X[:,1]
    temp_theta = theta

    print("shape of X is: ",X.shape)
    print("shape of y is: ",y.shape)
    print("shape of theta is: ", theta.shape)
    
    for i in range(no_of_iterations):
        hypothesis = np.matmul(X,theta)

        for j in range(no_of_features):
            derivative = (1/m)*sum((hypothesis-y)*X[:,j])
            temp_theta[j] = theta[j] - (alpha*derivative)
        theta = temp_theta       
        J_history[i] = computeCost(X,y,theta)
    return theta,J_history

###Normal Equation to return theta values
def normalEquation(X,y):
    print("\nshape of X: ",X.shape)
    print("\nshape of y: ",y.shape)
    
    XtransposeX = np.matmul(np.transpose(X),X)
##    print("\nshape of XtransposeX: ",XtransposeX.shape)
    Xtransposey = np.matmul(np.transpose(X),y)
##    print("\nshape of Xtransposey: ",Xtransposey.shape)
    theta = np.matmul(np.linalg.inv(XtransposeX),Xtransposey)
    return theta


if __name__ == '__main__':  
    Data=np.loadtxt('multiFeatureDataFile.txt',dtype=float,delimiter=',')
    X = Data[:,0:Data.shape[1]-1]
    y = Data[:,Data.shape[1]-1]
    m = y.shape[0]

    theta = np.zeros(X.shape[1]+1)

    X,mu,sigma=feature_normalize(X)

    #add intercept terms to X
    X=np.c_[np.ones(X.shape[0]),X]

    theta,J_values = gradientDescent(X,y,theta,alpha,iterations)

    print("\nTheta values found by gradient descent: ", theta)


    #pickling the theta values for future use
    with open('LinearRegression.pickle','wb') as f:
        pickle.dump(theta,f)

    pickle_in = open('LinearRegression.pickle','rb')
    theta = pickle.load(pickle_in)

    print("\npickled out theta values are : ", theta)

    print("computing theta using Normal equation: \n")
##    theta = normalEquation(X,y)
##
##    print("\nTheta values found using Normal equation: ",theta)

    #Plot the cost J with respect to no of iterations. We should be reducing the cost by a bit on every iteration
    iter=np.linspace(1,8000,num=8000,endpoint=False)
    plt.plot(iter,J_values,'-',color='r')
    plt.xlabel('Iterations')
    plt.ylabel('Cost J(theta)')
    plt.show(block=False)
