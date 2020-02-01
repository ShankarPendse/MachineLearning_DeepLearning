import time
start_time = time.time()
import numpy as np
import scipy.optimize as opt
import scipy.io as scipyio
import logisticRegression as lr
import logisticRegressionMultiClassClassification as multiclf
step_cost = []
##Function to compute and return the cost
def computeCost(nn_parameters,X,y,input_layer_size,hidden_layer_size,num_labels,lambda_value):
    
    Theta1 = nn_parameters[0:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_parameters[(hidden_layer_size*(input_layer_size+1)):].reshape(num_labels,hidden_layer_size+1)
    
    m = X.shape[0]
    J = 0

    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    I = np.identity(num_labels)
    Y = np.zeros((m,num_labels))

    for i in range(m):
        Y[i,:] = I[y[i]-1,:]

    layer1Activation = np.c_[np.ones(X.shape[0]),X]
    z2 = layer1Activation@Theta1.transpose()
    layer2Activation = lr.sigmoid(z2)
    layer2Activation = np.c_[np.ones(m),layer2Activation]
    z3 = layer2Activation@Theta2.transpose()
    hypothesis = lr.sigmoid(z3)

    squaredSumTheta = np.sum(np.square(Theta1[:,1:]))+np.sum(np.square(Theta2[:,1:]))
    
    regularizationTerm = (lambda_value/(2*m))*(squaredSumTheta)
    
    J = 0
    J = np.sum((Y*np.log(hypothesis))+((1-Y)*(np.log(1-hypothesis))))
    J = -(J/m) + regularizationTerm

    return J

##Function to compute and return gradients as a one dimensional vector
def computeGradient(nn_parameters,X,y,input_layer_size,hidden_layer_size,num_labels,lambda_value):

    Theta1 = nn_parameters[0:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_parameters[(hidden_layer_size*(input_layer_size+1)):].reshape(num_labels,hidden_layer_size+1)

    m = X.shape[0]
    J = 0

    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    I = np.identity(num_labels)
    Y = np.zeros((m,num_labels))

    for i in range(m):
        Y[i,:] = I[y[i]-1,:]

    layer1Activation = np.c_[np.ones(X.shape[0]),X]
    z2 = layer1Activation@Theta1.transpose()
    layer2Activation = lr.sigmoid(z2)
    layer2Activation = np.c_[np.ones(m),layer2Activation]
    z3 = layer2Activation@Theta2.transpose()
    hypothesis = lr.sigmoid(z3)

    delta3 = hypothesis - Y
    gprimez2 = sigmoidGradient(z2)

    delta2 = (delta3@Theta2[:,1:])*gprimez2

    gradient1 = delta2.transpose()@layer1Activation

    gradient2 = delta3.transpose()@layer2Activation

    Theta1_gradient = (1/m)*gradient1
    Theta2_gradient = (1/m)*gradient2

    regularizationTermForGradient1 = (lambda_value/m)*(np.sum(Theta1_gradient[:,1:]))

    regularizationTermForGradient2 = (lambda_value/m)*(np.sum(Theta2_gradient[:,1:]))

    Theta1_gradient = Theta1_gradient + regularizationTermForGradient1
    Theta2_gradient = Theta2_gradient + regularizationTermForGradient2

    Theta1_gradient = Theta1_gradient.reshape(Theta1_gradient.shape[0]*Theta1_gradient.shape[1],1)
    Theta2_gradient = Theta2_gradient.reshape(Theta2_gradient.shape[0]*Theta2_gradient.shape[1],1)

    gradient = np.r_[Theta1_gradient,Theta2_gradient]
    gradient = gradient.reshape(gradient.shape[0],1)

    return gradient.flatten()

def sigmoidGradient(z):
    g = np.zeros(z.shape)
    g = lr.sigmoid(z)*(1-lr.sigmoid(z))
    return g

def randInitialization(l_in,l_out):
    epsilon_init = 0.12
    Theta = np.zeros([l_out,l_in+1])
    Theta = np.random.rand(l_out,l_in+1)*(2*epsilon_init)-epsilon_init;
    return Theta

def predict(X,Theta1,Theta2):
    X = np.c_[np.ones(X.shape[0]),X]

    hiddenLayerOutput = lr.sigmoid(X@Theta1.transpose())
    hiddenLayerOutput = np.c_[np.ones(hiddenLayerOutput.shape[0]),hiddenLayerOutput]
    finalOutput = lr.sigmoid(hiddenLayerOutput@Theta2.transpose())
    predictions = np.argmax(finalOutput,axis=1)+1
    return predictions

if (__name__=='__main__'):

    Data = scipyio.loadmat('ex4data1.mat')

    input_layer_size  = 400
    hidden_layer_size = 25
    num_labels = 10
    lambda_value = 3
    X = Data['X']
    y = Data['y']

    parameters = scipyio.loadmat('ex4weights.mat')

    Theta1 = parameters['Theta1']
    Theta2 = parameters['Theta2']

    newTheta1 = Theta1.reshape(Theta1.shape[0]*Theta1.shape[1],1)
    newTheta2 = Theta2.reshape(Theta2.shape[0]*Theta2.shape[1],1)
    neuralNetworkParams = np.r_[newTheta1,newTheta2]

    print("\nTesting Regularized CostFunction, the value should be about 0.576051 for lambda = 3:")
    J = computeCost(neuralNetworkParams,X,y,input_layer_size,hidden_layer_size,num_labels,lambda_value)

    print("Cost is : ",J)

    initial_Theta1 = randInitialization(input_layer_size,hidden_layer_size)
    initial_Theta2 = randInitialization(hidden_layer_size,num_labels)

    gradient = computeGradient(neuralNetworkParams,X,y,input_layer_size,hidden_layer_size,num_labels,lambda_value)

    lambda_value = 1
    all_theta = opt.fmin_cg(computeCost,fprime = computeGradient,x0 = neuralNetworkParams,args=(X,y,input_layer_size,hidden_layer_size,num_labels,lambda_value))

    new_Theta1 = all_theta[0:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    new_Theta2 = all_theta[(hidden_layer_size*(input_layer_size+1)):].reshape(num_labels,hidden_layer_size+1) 

    predictions = predict(X,new_Theta1,new_Theta2)


    accuracy = multiclf.calculateAccuracy(predictions,y)
    print("Accuracy: ",accuracy,"%")
    
    print("\nIt took ",time.time()-start_time," seconds to complete the execution\n")

