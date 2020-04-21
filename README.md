# ML-Logistic-regression-algorithm-challenge

![DSN logo](DSN_logo.png)|DSN Algorithm Challenge|
|---|---|

Author: Adesola Tolulope Timilehin

Date: 21/04/2020

Building a logistic regresstion model from scratch:

The logistic regresstion model unlike the name suggests is a classification.
It is very similar to the linear Regression model.

The major difference is the sigmoid function; 
The sigmoid function is applied on the Linear regression function to change it from outputing continous values to probobalistic values.

# lr = the learning rate is usually very small, it determines how far we go in a direction with each step.
# n_iters = the number of iterations.
# weight = Weight is the strenght of the connection. its reflects the steepness of the sigmoid graph
# bias = Bias is how far away the real value a given value is.
           

# dw = This is the derivative w.r.t weight
# db = This is the derivative w.r.t bias

# y_pred = This is the predicted value/our y value 
# y_pred_classification = this will classify the predicted value into 1 or 0.

the predict function will return either 1 or 0 since our logistic regression model is a classification model.
We have succedded in turning our linear regression model which is continous into a classification model.

# LogisticRegression.ipynb: jupyter notebook script containing the code for building the model.

# LogisticRegression.py: python script containing the code for building the model.
  