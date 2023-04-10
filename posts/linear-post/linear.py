import numpy as np
from matplotlib import pyplot as plt

#To print out my file's name
def my_custom_function():
    print("I implemented this function in the file " + __name__ + ".py")

    
#LogisticRegression class
class LinearRegression:
    def __init__(self):
        # initialize instance variables
        self.w = 0
        self.score_history = []


    def pad(self, X):
        # helper function to ensure that X contains a column of 1s prior to any major computations
        return np.append(X, np.ones((X.shape[0], 1)), 1)
    

    def predict(self, X: np.matrix) -> np.ndarray:
        # predict function to predict the given X data
        return X@self.w


    def score(self, X: np.matrix, y: np.ndarray) -> int:
        # calculate the accuracy score of our prediction
        y_bar = np.mean(y)
        X = self.pad(X)        
        y_pred = self.predict(X)
        
        return 1 - np.sum((y_pred - y) ** 2)/np.sum((y_bar - y) ** 2)
    

    def fit(self, X: np.matrix, y: np.ndarray, method: str='analytic', alpha: float=0.001, max_epochs: int=1000) -> None:
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

        if method == 'analytic':
            self.w = np.linalg.inv(X_.T@X_)@X_.T@y
        else:
            # efficient way to calculate gradient
            P = X_.T@X_
            q = X_.T@y

            done = False
            prev_gradient = np.inf

            # initialize some random weights
            self.w = np.random.rand(X_.shape[1])
            
            for _ in range(max_epochs):
                # continue for max_steps or it's done
                if done:
                    break
                else:
                    gradient = 2*(P@self.w - q)

                    # update weights
                    self.w -= alpha*gradient

                    # update score history
                    self.score_history.append(self.score(X, y))

                    # check conditions
                    if np.allclose(gradient, prev_gradient):
                        done = True
                    else:
                        prev_gradient = gradient