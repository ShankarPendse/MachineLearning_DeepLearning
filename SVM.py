import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scipyio
import logisticRegression as lr
from sklearn import svm
start_time = time.time()

def visualize(X,y,clf):
    plt.figure(2)
    lr.plot(X,y)

##    X_min = X[:,0].min()
##    X_max = X[:,0].max()
##
##    y_min = X[:,1].min()
##    y_max = X[:,1].max()
##
##    h = (X_max/X_min)/100
##    
##    xx, yy = np.meshgrid(np.arange(X_min, X_max, h),np.arange(y_min, y_max, h))
##
##    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
##    Z = Z.reshape(xx.shape)
##    
##    plt.contour(Z)
##    
##    plt.show(block=False)


    x1plot = np.linspace(min(X[:,0]), max(X[:,0]), X.shape[0]).T

    x2plot = np.linspace(min(X[:,1]), max(X[:,1]), X.shape[0]).T

    X1, X2 = np.meshgrid(x1plot, x2plot)

    vals = np.zeros(X1.shape)



    for i in range(X1.shape[1]):

        this_X = np.column_stack((X1[:, i], X2[:, i]))

        vals[:, i] = clf.predict(this_X)


    plt.contour(X1, X2, vals, levels=[0.0, 0.0])

    plt.show(block=False)
    

   

if __name__ == '__main__':

##    Data = scipyio.loadmat('ex6data1.mat')
##    
##    X = Data['X']
##    y = Data['y']
##
##    print("size of X = ",X.shape)
##    print("size of y = ",y.shape)
##
##    plt.figure(1)
##    lr.plot(X,y)
##    
##    y = y.ravel()
##    
##    clf = svm.SVC(kernel = 'linear',C=100)
##    clf.fit(X,y)
##    
##    print(clf.predict(np.array([[2.016,3.318]])))
##    print("clf = ",clf)
##
##    theta = clf.coef_[0]
##
##    a = -theta[0] / theta[1]
##
##    plt.figure(2)
##    lr.plot(X,y)
##    yy = (a*X)-(clf.intercept_[0]/theta[1])
##    plt.plot(X,yy,color='b')
##    plt.show(block=False)
    
    Data = scipyio.loadmat('ex6data2.mat')
    
    X = Data['X']
    y = Data['y']

    print("size of X = ",X.shape)
    print("size of y = ",y.shape)

    plt.figure(1)
    lr.plot(X,y)

##    y = y.ravel()
    
    clf = svm.SVC(kernel = 'rbf',C=100)

    clf.fit(X,y.ravel())

    X_test = np.array([[1,2],[0.65,0.98],[5,6],[0.6,0.34]])
    y_predict = clf.predict(X_test)
    
    print("predictions for X_test are: ",y_predict)

    y_predict = clf.predict(X)
    y_predict = y_predict.reshape(y_predict.shape[0],1)
    y_predict = np.c_[y_predict,y_predict]

##    plt.show(block=False)
    visualize(X,y,clf)

    print("\nIt took ",time.time()-start_time," seconds to complete the execution\n")    
    
