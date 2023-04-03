import numpy as np
from sklearn.datasets import make_blobs
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

#To print out my file's name
def my_custom_function():
    print("I implemented this function in the file " + __name__ + ".py")

    
#LogisticRegression class
class LogisticRegression:
    def __init__(self):
        # initialize instance variables
        self.w = 0
        self.history = []
        self.loss_history = []
        

    def sigmoid(self, z):
        # helper function to calculate the sigmoid score
        return 1 / (1 + np.exp(-z))
    

    def pad(self, X):
        # helper function to ensure that X contains a column of 1s prior to any major computations
        return np.append(X, np.ones((X.shape[0], 1)), 1)


    def empirical_risk(self, X, y):
        # calculate the empirical risk given the weights on X and y
        y_hat = X@self.w
        return (-y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))).mean()
    

    def predict(self, X: np.matrix) -> np.ndarray:
        # predict function to predict the given X data
        return (X.dot(self.w) > 0) *1


    def score(self, X: np.matrix, y: np.ndarray) -> int:
        # calculate the accuracy score of our prediction
        return sum(np.equal(self.predict(X), y)) / len(y)
    

    def fit(self, X: np.matrix, y: np.ndarray, alpha: int, max_epochs: int) -> None:
        '''
        Has no return value. When it's called, it will train a model through
        gradient descent. It will update its weights based on inputs

            Parameters:
                    X (np.matrix): a nxp numpy matrix representing the sample data
                    y (np.ndarray): a numpy array representing the actual y value
                    alpha (int): the learning rate
                    max_epochs (int): the maximum number of iterations

            Returns:
                    none
        '''

        # ensure that X contains a column of 1s
        X_ = self.pad(X)

        done = False
        prev_loss = np.inf

        # initialize some random weights
        self.w = np.random.rand(X_.shape[1])
        
        for _ in range(max_epochs):
            # continue for max_steps or it's done
            if done:
                break
            else:
                # make predictions
                y_predict = X_@self.w 
                # calculate the gradient
                gradient = np.sum((np.multiply((self.sigmoid(y_predict) - y)[:, np.newaxis], X_)), axis=0)  / X_.shape[0]
                # update weights
                self.w -= alpha*gradient
                # calculate the new loss and make updating
                new_loss = self.empirical_risk(X_, y)
                self.loss_history.append(new_loss)
                self.history.append(self.score(X_, y))

                # check conditions
                if np.isclose(new_loss, prev_loss):
                    done = True
                else:
                    prev_loss = new_loss

    
    def fit_stochastic(self, X: np.matrix, y: np.ndarray, max_epochs: int, momentum: bool, batch_size: int, alpha: int) -> None:
        '''
        Has no return value. Another version of the fit method. When it's called, it will train a model through
        stochastic gradient descent. It will update its weights based on inputs

            Parameters:
                    X (np.matrix): a nxp numpy matrix representing the sample data
                    y (np.ndarray): a numpy array representing the actual y value
                    max_epochs (int): the maximum number of iterations
                    momentum (bool): indicates if we want to apply the momentum method
                    batch_size (int): the number observations in each batch
                    alpha (int): the learning rate
                    
            Returns:
                    none
        '''

        # ensure that X contains a column of 1s
        X_ = self.pad(X)
        n = X_.shape[0]

        # check momentum
        if momentum:
            momentum = 0.8
        else:
            momentum = 0

        done = False
        prev_loss = np.inf
        # initialize some random weights
        self.w = np.random.rand(X_.shape[1])

        
        for j in np.arange(max_epochs):
            # continue for max_steps or it's done
            if done:
                break
            else:
                # shuffle the data
                order = np.arange(n)
                np.random.shuffle(order)

                for batch in np.array_split(order, n // batch_size + 1):
                    x_batch = X_[batch,:]
                    y_batch = y[batch]
                    # make predictions
                    y_batch_predict = x_batch@self.w 
                    # calculate the gradient
                    # gradient = ((self.sigmoid(y_batch_predict) - y_batch)@x_batch)  / x_batch.shape[0] 
                    gradient = np.sum((np.multiply((self.sigmoid(y_batch_predict) - y_batch)[:, np.newaxis], x_batch)), axis=0)  / x_batch.shape[0]

                    # applying momentum formula
                    prev_w = self.w
                    self.w -= (alpha*gradient - momentum*(self.w - prev_w))

                    # calculate the new loss and make updating
                    new_loss = self.empirical_risk(X_, y)
                    self.loss_history.append(new_loss)
                    self.history.append(self.score(X_, y))

                    # check conditions
                    if np.isclose(new_loss, prev_loss):
                        done = True
                    else:
                        prev_loss = new_loss