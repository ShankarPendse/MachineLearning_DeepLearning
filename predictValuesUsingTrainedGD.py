import pickle
import numpy as np

Data=np.loadtxt('feauresForPrediction.txt',dtype=float,delimiter=',')

##X = Data[:,0:Data.shape[1]]
def feature_normalize(X):
    print("Normalizing the feaures")
    X_norm = X
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    for i in range(X_norm.shape[1]-1):
        X_norm[:,i] = (X_norm[:,i]-mu[i])/sigma[i]

    return X_norm,mu,sigma

pickle_in = open('LinearRegression.pickle','rb')
theta = pickle.load(pickle_in)
theta = theta[0]
mu = theta[1]
sigma = theta[2]

if(Data.ndim > 1):
    X = Data[:,0:Data.shape[1]]
    X,mu,sigma=feature_normalize(X)
    X = np.c_[np.ones(X.shape[0]),X]
else:
    for i in range(X.size):
        X[i] = (Data[i]-mu[i])/sigma[i]
    X = np.r_[np.ones(1),X]
    X = X[None,:]
##add intercept terms to X
#X = np.c_[np.ones(X.shape[0]),X]

pickle_in = open('LinearRegression.pickle','rb')
theta = pickle.load(pickle_in)
#theta = theta[:,None]
print("\npickled out theta values are : ", theta)

print("\nX = ",X)
print("\ntheta = ",theta)

#print("size of X and theta are: \n",X.shape,theta.shape)

y = np.matmul(X,theta)

print("\npredicted values are: ",y)
