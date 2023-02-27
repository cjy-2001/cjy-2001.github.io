import numpy as np
from sklearn.datasets import make_blobs
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

#To print out my file's name
def my_custom_function():
    print("I implemented this function in the file " + __name__ + ".py")

    
#Perceptron class
class Perceptron:
    def __init__(self):
        self.w = 0
        self.history = []
        
    
    
    def fit(self, X: np.matrix, y: np.ndarray, max_steps: int) -> None:
        #Fit function to train my model
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        self.w = np.random.rand(X_.shape[1])
        y_ = 2*y - 1 
        
        for _ in range(max_steps):
            #Continue for max_steps or the accuracy has reached to 1
            if self.score(X_, y_) == 1:
                break
            else:
                i = np.random.randint(0,X_.shape[0]-1)
                y_predict = self.predict(X_)
                self.w += (y_predict[i]*y_[i] < 0)*y_[i]*X_[i]
                self.history.append(self.score(X_, y_))

    
    def predict(self, X: np.matrix) -> np.ndarray:
        #Predict function to predict the given X data
        return np.sign(X.dot(self.w))


    def score(self, X: np.matrix, y: np.ndarray) -> int:
        #Calculate the accuracy score of our prediction
        return sum(np.equal(self.predict(X), y)) / len(y)