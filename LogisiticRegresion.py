import numpy as np

class LogisticRegression:
    def _init_(self, lr=0.001, n_iters=1000):     
        self.lr = lr               #lr is the learning rate                   
        self.n_iters = n_iters     #n_iters is the number of iterations
        self.weights = None        #the wieght
        self.bias = None           #the bias

    # The sigmoid function
    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def fit(self, X, y):
        #initializing parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0     
        
        for _ in range(self.n_iters):

            #the linear Regreession model
            linearRegression_model = np.dot(X, self.weights) + self.bias

            #Applying the sigmond function on the linear regression model
            y_pred = self._sigmoid(linearRegression_model)
            

            #Applying the update rule
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linearRegression_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linearRegression_model)
        y_pred_classification = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_classification