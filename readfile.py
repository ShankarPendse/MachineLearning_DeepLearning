##from numpy import *
##import math

import numpy as np


##y = np.array([[10],[10],[3],[3],[3],[5]])
##no_of_labels = 3
##rows = 6
##for i in range(1,no_of_labels+1):
##    y_i = np.array([1 if label==i else 0 for label in y])
####    y_i = np.reshape(y_i,(rows,1))
##    print(y_i)

y = np.array([[10],[10],[3],[3],[3],[5]])
no_of_labels = 3
rows = 6

for i in range(1,no_of_labels+1):
    for label in y:
        if label == i:
            y_i = np.array([1])
        else:
            y_i = np.array([0])
print(y_i)

##y_i = np.array([1,1,0,1,1,1])
##print(y_i)
####y_i = np.reshape(y_i,(6,1))
####print(y_i)

##data = np.loadtxt('ex2data2.txt',dtype=float,delimiter=',')
##
##
####data = genfromtxt('ex2data2.txt',dtype=float,delimiter=',')
##X = data[:,0:data.shape[1]-1]
##y = data[:,data.shape[1]-1]
##y = y.reshape(y.shape[0],1)
##
##X1 = X[:,0]
##X2 = X[:,1]
##
##out = np.ones(X1.shape[0])
##out = out.reshape(out.shape[0],1)
####out = ones((shape(X1)[0],1))
####
####print("shape of out is : ",out.shape)
##
##print(out.shape)
##degrees = 6
##for i in range(1, degrees+1):
##
##		for j in range(0, i+1):
##
##			term1 = X1 ** (i-j)
##
##			term2 = X2 ** (j)
##
##			term  = (term1 * term2).reshape( term1.shape[0], 1 ) 
##
##			out   = np.hstack(( out, term ))
##
##print("\nmin = ",np.amin(out[:,1])-2)
##print("\nmax = ",np.amax(out[:,1])+2)
##degree = 6
##out1 = ones(X1.shape[0])
##out1 = out.reshape(out.shape[0],1)
##print(out1.shape)
####print(out.ndim)
####
####print(out[:,0])
##
####print(out.shape)
##
##for i in range(degree):
##    for j in range(i):
##        out = np.c_[out,np.power(X1,(i+1)-j) * np.power(X2,j)]
##
##X = out
##theta = np.c_[np.ones(X.shape[1])]
##print("theta\n")
##print("\nshape: ",theta.shape)
##print("\ndimension: ",theta.ndim)
##
##print(theta[1:])
##print("\n-----------\n")
##
##sumOfTheta = sum(np.power(theta[1:],2))
##
##print("sum = ",sumOfTheta)
##
##
##grad = np.zeros(theta.shape)
##print("grad shape: ",grad.shape)
##print("grad first value: ",grad[15,0])

##print("\n shpae of out is : ",out.shape)
##print("\n first row of out is : ",out[0,:])

##for i in range(degree):
##    for j in range(i):
##        out[:,i+1] = np.power(X1,(i+1)-j) * np.power(X2,j)




##print(data.shape)
##print(type(data))

##A = np.matrix(np.ones(4))

####A = np.array[("1;2;3")]
####print(A)

##A = np.zeros(3)
##print(A[0])

##A = np.matrix("1;2;3;4;5;6")
##print(A)

##X = data[:,0:data.shape[1]-1]
##print("\nsize of X = ",X.shape)
##
##y=data[:,data.shape[1]-1]
##print("\nsize of y = ",y.shape)
##
##theta = np.zeros(X.shape[1])
##print("\ntheta values = ",theta)
##print("\nsize of theta = ",theta.shape)




##file=open('ex1data1.txt','r')
##content=file.read()
##
###print(content)
###print(len(content))
##
##X=np.loadtxt('ex1data1.txt',dtype=float,delimiter=',')
##print(X.shape)

###print(content)
##print("content is of type: ",type(content))
##print("length of content is: ", len(content))
###print(content(2))
##X=np.matrix(content)
##Y=np.transpose(X)
##
##print("Size of X is: ",X.shape)
##print("Size of Y is: ",Y.shape)

##print("X is of type: ",type(X))
##print("size of X is: ", X.shape)

##data = np.loadtxt('ex1data1.txt',dtype=float,delimiter=',')
##X=data[:,0]
##y=data[:,1]
##
##print(X.shape)
##print(y.shape)

#print(data[0,:])

##A=data[0:5,:2]
##B=data[6:11,:2]
##print(A)
##print(B)
##C=A*B
##print(C)
 
#A=np.array([[1,2],[3,4],[5,6]])
##A=np.array([[1],[2],[3]])
##B=np.array([[1,2],[3,4],[5,6]])
##
##print("\nA = ",A)
##print("\nType of A is: ", type(A))
##
##transposeOfA = np.transpose(A)
##print("\n Transpose of A is = ",transposeOfA)
##
##print(np.transpose(transposeOfA))

#Anew=np.c_[np.ones(A.shape[0]),A]
##sumofA=sum(A)

##print("\nsum of A is = ",np.sum(A))
#print("\nA = ",A)
#print("\nA= ",Anew)
####print("\nB= ",B)
####
####
####print("\nmatrix multiplication = ",np.matmul(Anew,B))
####print("\nelementwise multiplication = ", A*B)
####
####
####y=A[:,A.shape[1]-1]
####y=y[:,None]
####print("\ny= ", y)
##
##C=A+B

##print("\nB = ",B)
##print("\nC = ",C)
##mu=np.mean(A,axis=0)
##print(mu)
##print(mu[0])
##print(mu[1])
##print("\nmean of A column wise is = ",np.mean(A,axis=0))
##print("\nstandard deviation of A column wise is = ",np.std(A,axis=0))
