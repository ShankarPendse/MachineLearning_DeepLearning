#####Logistic Regression using advanced optimization algorithm TNC#####
#####__________________________________________________________#####

import time
start_time = time.time()
import numpy as np
import math
import scipy.optimize as opt
from matplotlib import pyplot as plt

###Function to plot the initial datapoints
def plot(X,y):
    pos_examples = np.where(y==1)
    neg_examples = np.where(y==0)
    plt.scatter(X[pos_examples,0],X[pos_examples,1],marker = '+',color = 'black')
    plt.scatter(X[neg_examples,0],X[neg_examples,1],marker = '.',color = 'yellow')
    plt.show(block=False)

###Function to plot decision boundary
def plotDecisionBoundary(X,y,theta):
    plot(X[:,1:3],y)
    print("\ntheta values = ",theta)
    if(X.shape[1] <= 3):
        plot_x = np.array([min(X[:,1]), max(X[:,1])])
        plot_y = (-1./theta[2])*(theta[1]*plot_x+theta[0])
        plt.plot(plot_x,plot_y)
        
##Function to compute the cost of logistic regression, which is: J = -1/m*(ylog(hypothesis)+(1-y)log(1-hypothesis))
def costFunction(theta,X,y):
    m = len(y)
    hypothesis = sigmoid(X@theta)
    log_of_hypothesis = np.log(hypothesis)
    log_of_oneMinusHypothesis = np.log(1-hypothesis)

    J = -(1/m)*(((y.transpose()@log_of_hypothesis))+((1-y.transpose())@log_of_oneMinusHypothesis))
    return np.asscalar(J)

##Regularized cost function
def regularized_costFunction(theta,X,y,lambdaValue):
    m = len(y)
    if(theta.ndim == 1):
        theta = theta.reshape(theta.shape[0],1)
    hypothesis = sigmoid(X@theta)
    log_of_hypothesis = np.log(hypothesis)
    log_of_oneMinusHypothesis = np.log(1-hypothesis)
    regularizationTerm = (lambdaValue/(2*m)) * sum(np.power(theta[1:],2))
    
    J = -(1/m)*(((y.transpose()@log_of_hypothesis))+((1-y.transpose())@log_of_oneMinusHypothesis)) + regularizationTerm
    return np.asscalar(J)

##Regularized gradient computation
def regularized_gradient(theta,X,y,lambdaValue):
    m = len(y)
    if(theta.ndim == 1):
        theta = theta.reshape(theta.shape[0],1)
    grad = np.zeros(theta.shape)
    hypothesis = sigmoid(X@theta)
    error = hypothesis-y
    regularizationTerm = (lambdaValue/m)*theta
    grad = ((1/m)*(X.transpose()@error)) + regularizationTerm
    grad[0,0] = grad[0,0]-regularizationTerm[0,0] ##should not be regualrizing for theta[0]
    if (__name__ == '__main__'):
        return grad
    else:
        return grad.flatten()

##Function to compute gradient which is : grad = (1/m)*(hypothesis-y)*X[:,j]
def gradient(theta,X,y):
    m = len(y)
    hypothesis = sigmoid(X@theta)
    error = hypothesis-y
    grad = (1/m)*(X.transpose()@error)
    return grad

###Function to compute sigmoid for given values
def sigmoid(z):
    g=(1/(1+np.exp(-z)))
    return g

###Function to predict the accuracy of the training
def predict(X,theta):
    m = X.shape[0]
    p = np.zeros(X.shape[0])
    hypothesis = sigmoid(X@theta)
    p[np.where(hypothesis >= 0.5)] = 1
    return p

###Function to calculate the accuracy after prediction
def calculateAccuracy(predictedValues,actualValues):
    m = len(actualValues)
    accuracy = 0
    True_positive = 0
    True_negative = 0
    for i in range(m):
        if (predictedValues[i] == 0 and actualValues[i] == 0):
            True_negative+= 1
        elif(predictedValues[i] == 1 and actualValues[i] == 1):
            True_positive+= 1
    accuracy = (True_positive + True_negative)*(100/m)
    return accuracy

###Function for Feature mapping
def mapFeature(X1,X2):
    degree = 6
    out = np.ones(X1.shape[0])
    out = out.reshape(out.shape[0],1)
    for i in range(1,degree+1):
        for j in range(i+1):
            out = np.c_[out,np.power(X1,i-j) * np.power(X2,j)]
    print("shape of out: ",out.shape)
    return out

if __name__ == '__main__':

    initial_lambda = 1

    print("\nlogistic regression model to predict whether a student gets admitted into a university")

    ###Load the data file for training which contains the 100 examples of students with marks in two exams as first two columns and third column eithre 0 or 1, 0: no admission, 1: admission

    data = np.loadtxt('ex2data1.txt',dtype=float,delimiter=',')

    X = data[:,0:data.shape[1]-1]
    y = data[:,data.shape[1]-1]
    y = y.reshape(y.shape[0],1)

    ##plot(X,y)
    ##plt.xlabel('Exam1 Score')
    ##plt.ylabel('Exam2 Score')
    ##plt.show(block=False)

    X = np.c_[np.ones(X.shape[0]),X]

    theta = np.c_[np.zeros(X.shape[1])]

    test_theta = np.array([[-24],[0.2],[0.2]])

    print("\nComputing Cost and gradient with theta values set to 0")
    cost = costFunction(theta,X,y)

    print("\nCost is computed\n\ncomputing gradient now")
    grad = gradient(theta,X,y)

    print("\ngradient computed and the values are: ")

    print("\nCost = ",cost)
    print("\ngradient = ",grad)

    print("\nComputing Cost and gradient using non zero theta values")
    cost = costFunction(test_theta,X,y)

    print("\nCost is computed\n\ncomputing gradient now")
    grad = gradient(test_theta,X,y)

    print("\ngradient computed and the values are: ")

    print("\nCost = ",cost)
    print("\ngradient = ",grad)


    print("\nRunning advanced optimization algorithm TNC")

    #Result = opt.minimize(costFunction, theta,args=(X,y),jac=False,method='TNC',options={'maxiter':136})
    Result = opt.minimize(costFunction, theta,args=(X,y),jac=False,method='TNC')

    print("\nTNC Algorightm execution completed")

    cost = Result['fun']
    gradient = Result['jac']
    theta = Result['x']

    print("\nvalues obtained using advanced optimization algorithm TNC: \n")
    print("cost: ",cost)
    print("gradient: ",gradient)
    print("theta: ",theta)

    print("\nPredicting for new values of X [1,45,85]")
    newX = [1,45,85]
    prob = sigmoid(newX@theta)

    print("\nprobability of getting admission is : ",prob*100,"%")

    predictedValues = predict(X,theta)

    print("\ncomputing accuracy of the model\n")

    accuracyPercent = calculateAccuracy(predictedValues,y)
    print("Accuracy of the training model is: ", accuracyPercent,"%")

    print("\nPlotting decision boundary:\n")

    plotDecisionBoundary(X,y,theta)
    plt.show(block=False)

    print("\n-------------------------------------------------------------------------------------------------------------\n")

    print("\logistic regression model to predict whether microchips from a fabrication plant passes quality assurance (QA)")

    data = np.loadtxt('ex2data2.txt',dtype=float,delimiter=',')

    X = data[:,0:data.shape[1]-1]
    y = data[:,data.shape[1]-1]
    y = y.reshape(y.shape[0],1)

    ##plot(X,y)
    ##plt.xlabel('Microchip Test 1')
    ##plt.ylabel('Microchip Test 2')
    ##plt.show(block=False)

    X = mapFeature(X[:,0],X[:,1])

    theta = np.c_[np.zeros(X.shape[1])]

    cost = regularized_costFunction(theta,X,y,initial_lambda)
    grad = regularized_gradient(theta,X,y,initial_lambda)

    print("\ncost computed is: ",cost)
    print("\nfirst 5 Grad computed is: ",grad[0:5,:])

    test_theta = np.c_[np.ones(X.shape[1])]
    test_lambda = 10

    cost = regularized_costFunction(test_theta,X,y,test_lambda)
    grad = regularized_gradient(test_theta,X,y,test_lambda)

    print("\nwith thetas initialized to 1 and lambda = 10")

    print("\ncost computed is: ",cost)
    print("\nfirst 5 Grad computed is: ",grad[0:5,:])

    print("\nRunning TNC algo to optimize")
    Result = opt.minimize(regularized_costFunction, theta,args=(X,y,initial_lambda),jac=False,method='TNC')
    print(Result)

    cost = Result['fun']
    gradient = Result['jac']
    theta = Result['x']

    predictedValues = predict(X,theta)

    accuracyPercent = calculateAccuracy(predictedValues,y)
    print("\nAccuracy of the training model is: ", accuracyPercent,"%")

    print("\nIt took ",time.time()-start_time," seconds to complete the execution\n")
