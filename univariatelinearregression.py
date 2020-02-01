import numpy as np
import matplotlib.pyplot as plt

#set number of iterations and learning rate for gradient descent
iterations=1500
alpha=0.01

#Load the data from the data file as m*n array
Data=np.loadtxt('unifeatureDataFile.txt',dtype=float,delimiter=',')

xs = Data[:,0]
ys = Data[:,1]

#Get the X,y and theta matrices
X = np.matrix(np.c_[np.ones(xs.shape[0]),xs])
y = np.matrix(ys).transpose()
theta = np.matrix(np.zeros(2)).transpose()

print("X shape: ",X.shape)
print("y shape: ", y.shape)

#Compute the cost (mean squared error)
def computeCost(X,y,theta):
    m = y.shape[0]
    hypothesis = X*theta
    errors = hypothesis-y
    squared_errors = np.multiply(errors,errors)
    sum_squared_errors = np.sum(squared_errors)
    J = sum_squared_errors/(2*m)
    return J

#Gradient descent algorithm
def gradientDescent(X,y,theta,alpha,no_of_iterations):
    m = y.shape[0]
    J_history = np.matrix(np.zeros(no_of_iterations)).transpose()

    featurevalue = X[:,1]

    for i in range(no_of_iterations):
        hypothesis = theta[0,0] + (theta[1,0]*featurevalue)
        temp0 = theta[0,0] - (alpha*(1/m)*sum(hypothesis-y))
        temp1 = theta[1,0] - (alpha*(1/m)*sum(np.multiply((hypothesis-y),featurevalue)))

        theta[0,0] = temp0
        theta[1,0] = temp1

        J_history[i]=computeCost(X,y,theta)

    return theta,J_history


print("\n Testing the cost function\n")

J = computeCost(X,y,theta)
print("with theta = [0;0]\ncost computed is: ",J)

J = computeCost(X,y,np.matrix('-1;2'))
print("further testing of cost function with theta [-1;2]\ncost computed is: ",J)

theta,J_values = gradientDescent(X,y,theta,alpha,iterations)

print("\nTheta values found by gradient descent: ", theta)

#Plot the cost J with respect to no of iterations. We should be reducing the cost by a bit on every iteration
iter=np.linspace(1,1500,num=1500,endpoint=False)
plt.figure(1)
plt.scatter(X[:,0].flatten(),y)
plt.show(block=False)
plt.figure(2)
plt.plot(iter,J_values,'-',color='r')
plt.xlabel('Iterations')
plt.ylabel('Cost J(theta)')
plt.show(block=False)
