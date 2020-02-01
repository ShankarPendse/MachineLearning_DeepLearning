import scipy.io as scipyio
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


## Function to calculate regularized Cost function. Can be also non regularized if lambdaValue is set to 0
def regularizedCost(theta,X,y,lambdaValue):
    m = X.shape[0]
    if(theta.ndim == 1):
        theta = theta.reshape(theta.shape[0],1)
    hypothesis = X@theta
    squaredError = np.sum(np.square(hypothesis-y))
    regularizationTerm = (lambdaValue/(2*m))*np.sum(np.square(theta[1:,:]))
    cost = ((1/(2*m))* squaredError)+regularizationTerm
    return cost

## Function to calculate Gradients using regularization. Can be also non regularized if lambdaValue is set to 0
def computeGradient(theta,X,y,lambdaValue):
    m = X.shape[0]
    if(theta.ndim == 1):
        theta = theta.reshape(theta.shape[0],1)
    gradients = np.zeros(theta.shape[0])
    hypothesis = X@theta
    hypothesis = hypothesis.reshape(hypothesis.shape[0],1)
    
    error = hypothesis-y
    
    regularizationTerm = (lambdaValue/m)*theta
    gradients = ((1/m)*(X.transpose()@error)) + regularizationTerm
    gradients[0,0] = gradients[0,0] - regularizationTerm[0,0]

    return gradients.flatten()

##Function to train the model using fmin_cg algorithm to obtain optimal Theta values
def trainLinearRegression(X,y,lambdaValue):
    initial_theta = np.zeros(X.shape[1])
    initial_theta = initial_theta.reshape(initial_theta.shape[0],1)
    
    theta = opt.fmin_cg(regularizedCost,fprime = computeGradient,x0 = initial_theta,args = (X,y,lambdaValue))
    return theta

##Function to calculate the train error and CV error
def learningCurve(X,y,xval,yval,lambdaValue):
    m = X.shape[0]
    trainError = np.zeros(m)
    cvError = np.zeros(m)
    for i in range(1,m+1):
        theta = trainLinearRegression(X[0:i,:],y[0:i],lambdaValue)
        Jtrain = regularizedCost(theta,X[0:i,:],y[0:i],lambdaValue)
        Jcv = regularizedCost(theta,xval,yval,lambdaValue)
        trainError[i-1] = Jtrain
        cvError[i-1] = Jcv
    return trainError,cvError

##Function to compute train error and validation error for different values of lambda
def validationCurve(X_poly,y,Xval_poly,yval):
    lambdaValues = np.array([0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10])
    error_train = np.zeros(len(lambdaValues))
    error_val = np.zeros(len(lambdaValues))
    theta = np.zeros(np.array((len(lambdaValues),X_poly.shape[1])))
    for i in range(len(lambdaValues)):
        lambdaValue = lambdaValues[i]
        theta[i,:] = trainLinearRegression(X_poly,y,lambdaValue)
        Jtrain = regularizedCost(theta[i,:],X_poly,y,0)
        Jcv = regularizedCost(theta[i,:],Xval_poly,yval,0)
        error_train[i] = Jtrain
        error_val[i] = Jcv
    return theta,lambdaValues,error_train,error_val

##Function to add more features (obtain polynomials)
def polyFeature(X,p):

    X_poly = np.zeros((X.shape[0],p))

    for i in range(1,p+1):
        X_poly[:,i-1] = X[:,0]**i
    return X_poly

##Normalize the features value
def featureNormalize(X):
    X_norm = X
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    for i in range(X_norm.shape[1]):
        X_norm[:,i] = (X_norm[:,i]-mu[i])/sigma[i]

    return X_norm,mu,sigma

##Function to plot the polynomial featured values
def plotFit(min_x,max_x,mu,sigma,theta,p):
    x = np.arange(min_x-15,max_x+25)
    x = x.reshape(x.shape[0],1)
    X_poly = polyFeature(x, p);
    for i in range(X_poly.shape[1]):
        X_poly[:,i] = (X_poly[:,i]-mu[i])/sigma[i]
    X_poly = np.c_[np.ones(X_poly.shape[0]),X_poly]
    plt.plot(x,X_poly@theta,color='b')

##def findMinJcv(error_val,theta):
    

if (__name__ == '__main__'):
    
    data = scipyio.loadmat('ex5data1.mat')
##    lambdaValue = 3
    
    yval = data['yval']
    Xtest = data['Xtest']
    X = data['X']
    Xval = data['Xval']
    y = data['y']
    ytest = data['ytest']

    m = X.shape[0]
    
    plt.figure(1)
    plt.scatter(X,y,marker='x',color='r')
    plt.xlabel('Change in water level(X)')
    plt.ylabel('Water flowing out of the dam(y)')
    plt.title('Figure 1: Linear Fit')

    theta = np.array([[1],[1]])
    cost  = regularizedCost(theta,np.c_[np.ones(X.shape[0]),X],y,1)
    gradients = computeGradient(theta,np.c_[np.ones(X.shape[0]),X],y,1)
    print("cost: ",cost)
    print("gradients: ",gradients)

    lambdaValue = 0
    
    theta = trainLinearRegression(np.c_[np.ones(X.shape[0]),X],y,lambdaValue)

    theta = theta.reshape(theta.shape[0],1)

    plt.plot(X,np.c_[np.ones(X.shape[0]),X]@theta,color='b')



    
    error_train,error_val = learningCurve(np.c_[np.ones(X.shape[0]),X],y,np.c_[np.ones(Xval.shape[0]),Xval],yval,lambdaValue)

    plt.figure(2)
    plt.plot(range(1,m+1),error_train,label='Jtrain')
    plt.plot(range(1,m+1),error_val,label='Jcv')
    plt.axis([0,14,0,160])
    plt.legend()
    plt.title('Figure 2: Linear Regression learning curve without polynomial')
    plt.xlabel('number of training examples')
    plt.ylabel('error')
    
    p = 8

    X_poly = polyFeature(X,p)
    X_poly,mu,sigma = featureNormalize(X_poly)
    X_poly = np.c_[np.ones(X_poly.shape[0]),X_poly]

    theta = trainLinearRegression(X_poly,y,lambdaValue)

    plt.figure(3)
    plt.scatter(X,y,marker='x',color='r')
    plt.axis([-80,80,-60,100])
    
##    plotFit(X,np.amin(X),np.amax(X),mu,sigma,theta,p)

    plotFit(np.amin(X),np.amax(X),mu,sigma,theta,p)
    
    plt.xlabel('Change in water level(X)')
    plt.ylabel('Water flowing out of the dam(y)')
    plt.title('Figure 3: Polynomial Fit')

    Xval_poly = polyFeature(Xval,p)
    Xval_poly,mu,sigma = featureNormalize(Xval_poly)
    Xval_poly = np.c_[np.ones(Xval_poly.shape[0]),Xval_poly]

    Xtest_poly = polyFeature(Xtest,p)
    Xtest_poly,mu,sigma = featureNormalize(Xtest_poly)
    Xtest_poly = np.c_[np.ones(Xtest_poly.shape[0]),Xtest_poly]
    
##    error_train,error_val = learningCurve(np.c_[np.ones(X_poly.shape[0]),X_poly],y,np.c_[np.ones(Xval_poly.shape[0]),Xval_poly],yval,lambdaValue)
    error_train,error_val = learningCurve(X_poly,y,Xval_poly,yval,lambdaValue)
    plt.figure(4)
    plt.plot(range(1,m+1),error_train,label='Jtrain')
    plt.plot(range(1,m+1),error_val,label='Jcv')
    plt.axis([0,13,0,100])
    plt.legend()
    plt.xlabel('number of training examples')
    plt.ylabel('error')
    plt.title('Figure 4: Polynomial Learning curve')
    

    theta,lambdaValues,error_train,error_val = validationCurve(X_poly,y,Xval_poly,yval)

    plt.figure(5)
    plt.plot(lambdaValues,error_train,label='Jtrain')
    plt.plot(lambdaValues,error_val,label='Jcv')
##    plt.axis([0,10,0,20])
    plt.legend()
    plt.xlabel('lambda')
    plt.ylabel('error')
    plt.title('Figure 5: Selecting lambda using cross validation set')
    plt.show(block=False)

##    minTheta,minJcv = findMinJcv(error_val,theta)
    minJcv = np.amin(error_val)
    minIndex = np.argmin(error_val)
    lambdaValue = lambdaValues[minIndex]
    minTheta = theta[minIndex]
    print("Min validation error is: ",minJcv," and the relative theta values are : ",theta[minIndex]," For lambda = ",lambdaValue)
    testError = regularizedCost(theta[minIndex],Xtest_poly,ytest,lambdaValue)
    print("Test set error: ",testError)


    print("values obtained from validation curve function: \n")
    print("lambdaValues = ",lambdaValues)
    print("Error Train = ",error_train)
    print("Error val = ",error_val)
    
    plt.figure(6)
    plt.scatter(Xtest,ytest)
    plotFit(np.amin(Xtest),np.amax(Xtest),mu,sigma,minTheta,p)
##    plt.plot(Xtest_poly,Xtest_poly@minTheta,color = 'black',label='hypothesis')
    plt.xlabel('Change in water level(Xtest)')
    plt.ylabel('Water flowing out of the dam(ytest)')
##    plt.legend()
    plt.title('Figure 6: polyFit for test set with theta values which gives minimum validation error')
    plt.show(block=False)

    print(regularizedCost.__name__)
    print(regularizedCost.__doc__)

    
