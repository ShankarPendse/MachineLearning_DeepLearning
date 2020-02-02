1. compute cost and gradient using cost function for regularized linear regression with lambda = 1 and theta values initialezed to 1
2. compute theta values using fmin_cg advanced minimization algorithm with lambda = 0
3. plot the figure to check how the linear regression is fitting the data with the theta values obtained from fmin_cg
4. we can see that the line fit is not a perfect one for the data we are trying to fit
5. To understand if there is a high bias(under fitting) or high variance (over fitting) plot the learning curve using X,y,Xval and yval with lambda = 0
	learning curve:
					for each training example: (1 to m)
		
						1. Find out theta values using fmin_cg advanced minimization algorithm with lambda = 0
						2. compute the cost for that training example using the cost function for regularized linear regression
						3. compute the cost for the Cross validation set(entire data) using the cost function for regularized linear regression
						4. save the error obtained at step 4 and step 5 in separate arrays as train_error and validation_error

6. plot the train_error and validation_error against the number of examples on X axis
7. From the graph, we can see that as the number of training examples increases the train error also increases and also the validation error is not minimal
8. from the graph we can conclude that our hypothesis is not fitting the training data properly (underfitting the data)
9. To overcome under fitting of the data, we have to include more features (use polynomial regression instead of with only one feature)
10. construct a new data set using the existing data set such that, each column of the new data set represents the powered value to its column number (x1^1,x1^2,x1^3 etc till power 8)
11. convert the test data and validation data as well similar to the train data set
12. since the data varies exponentially when compared to all the columns, we have to normalize the data using feature normalization method, so that all the data values of all the columns are in same range, feature normalization is a function which returns normalized features data set, mu and sigma value
13. compute theta values using fmin_cg advanced minimization algorithm with lambda = 0 and the newly constructed training data with polynomial degree
14. plot original data X and output y along x and y axis respectively and fit the hypothesis, which is calculated using Xpoly*theta (theta is the obtained using the fmin_cg function at step 13)
15. Add 1's column to polynomial features data for training data set , test data set and validation set
16. plot learning curve using Xpoly,y,XvalPoly,yval with lambda = 0
17. plot the train error and validation error obtained from the learning curve agains the number of examples on X axis
18. From the learning curve we came to know that the polynomial regression is overfitting the data as train error is always 0 but validation error is high
19. To over come overfitting of the data, we need to regularize the theta associated with higher order polynomials. To do this, we have to increase the lambda value (choose the suitable lambda)
20. we can choose a sutiable lambda value using validation curve
	validation curve:
					for each lambda value in a lambda vector:
						1. Find out theta values using fmin_cg advanced minimization algorithm for Xpoly and y
						2. compute the cost for the training set Xpoly using the cost function for regularized linear regression using the theta values obtained from previous step
						3. compute the cost for the validation set Xpoly_val using the cost function for regularized linear regression using the same theta values used in previous step
						4. save the training set errors, validation set errors and theta values in three different arrays as train_error, validation_error and theta
21. Plot train_error and validation_error against lambda values in lambda vector against X axis
22. from the validation_error array pick the one with low error, respective theta from the theta array and lambda value from the lambda vector
23. test the theta values with the test data set that we have not used till now to see how our polynomial regression fits the data and compute the test error using cost function for regularized linear regression with the lambda value choosen from the previous step (which gave min error for validation set)



Note: 1's column is not added to the original data set and polynomial data sets till step 14 explicitly
