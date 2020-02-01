import time
start_time = time.time()
import scipy.io as scipyio
import numpy as np
import scipy.optimize as opt
import logisticRegressionMultiClassClassification as multiclf
import logisticRegression as lr
import numpy.random as nprandom

def predict(Theta1,Theta2,X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    predictions = np.zeros((X.shape[0],1))
    layer1Activation = np.c_[np.ones(X.shape[0]),X]
    z2 = layer1Activation@Theta1.transpose()
    layer2Activation = lr.sigmoid(z2)
    layer2Activation = np.c_[np.ones(m),layer2Activation]
    z3 = layer2Activation@Theta2.transpose()
    hypothesis = lr.sigmoid(z3)
    predictions = np.argmax(hypothesis,axis=1)+1
    return predictions


if (__name__ == '__main__'):
    
    print("\nloading data...")
    Data = scipyio.loadmat('ex3data1.mat')
    Parameters = scipyio.loadmat('ex3weights.mat')
    X = Data['X']
    y = Data['y']
    m = X.shape[0]
    theta1 = Parameters['Theta1']
    theta2 = Parameters['Theta2']
    print("loading complete")

    predictions = predict(theta1,theta2,X)

    accuracy = multiclf.calculateAccuracy(predictions,y)

    print("Accuracy during training the neural network is : ",accuracy,"%")

    rp = nprandom.permutation(m)
    count = 0
    no_of_correct_predictions = 0
    index_of_incorrect_predictions = list()
    failed_prediction_values = list()
    for i in range(m):
        x = X[rp[i],:]
        x = x.reshape(x.shape[0],1)
        pred = predict(theta1,theta2,x.transpose())
        
        count+=1

        if (y[rp[i]] == pred):
            no_of_correct_predictions+=1
        else:
            index_of_incorrect_predictions.append(i)
            failed_prediction_values.append(pred)


    print("Accuracy during prediction is: ",(no_of_correct_predictions/count)*100,"%")
##    i = 0
##    for index in index_of_incorrect_predictions:
##        print("predicted value is : ",failed_prediction_values[i]," for Actual Value of ",y[index])
##        i+=1

    print("\nIt took ",time.time()-start_time," seconds to complete the execution\n")
    
