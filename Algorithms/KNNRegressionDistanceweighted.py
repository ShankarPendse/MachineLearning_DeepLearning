import numpy as np
import time

# Reading the train and test data
train_data = np.genfromtxt('Regression/trainingData.csv', delimiter = ",")
test_data = np.genfromtxt('Regression/testData.csv', delimiter = ",")

# calculates and returns the euclidean distance between 1 query instance and all of the training instances
def calculate_distances(training_features, query_features):
    return np.sqrt(np.sum(np.square(query_features - training_features), axis = 1))
    
def predict(training_features, query_features):
    ''' training_features is a 2D numpy array, query_features is a single row from test data (1D numpy array)
        
            1) This function will first call calculate_distances to get the distance between a query instance
            and all of the training feature instances
            2) Select 3 nearest training feature instances (as 3 nearest neighbours)
            3) Calculate the average of the target feature of selected 3 neighbours with weights assigned to them as 
               (1/distance), where distance is the euclidean distance returned by calculate_distnaces function
            4) Return the calculated average as the predicted target value for the query instance'''
    
    distances = calculate_distances(training_features, query_features)
    
    # Set the number of nearest neighbours to consider
    k = 3 # We can paramterize this so that while calling this function, value of K can be passed as an argument
    
    # use np.argsort to get the indices of the values in sorted order
    indices = np.argsort(distances)
    
    # average the 3 nearest disances
    #predicted_value = np.mean(train_data[indices[0:k],-1])
    
    # Inverse distance weighted average of 3 nearest neighbours
    predicted_value = np.sum(train_data[indices[0:k],-1] * (1/(distances[indices[0:k]]))) / np.sum((1/(distances[indices[0:k]])))
    
    # squared Inverse distance weighted average of 3 nearest neighbours
    #predicted_value = np.sum(train_data[indices[0:k],-1] * np.square(1/(distances[indices[0:k]]))) / np.sum(np.square(1/(distances[indices[0:k]])))
    
    # return the prediction
    return predicted_value

def calculate_r2(actual_target_values, predicted_target_values):
    ''' This function takes the actual target values and predicted target values as arguments,
        returns the R squared score'''
    
    sum_squared_residuals = np.sum(
                                   np.square(predicted_target_values - actual_target_values)
                                  )
    
    sum_squares = np.sum(
                         np.square(np.mean(actual_target_values) - actual_target_values)
                        )
    
    r2 = 1 - (sum_squared_residuals/sum_squares)
    return r2