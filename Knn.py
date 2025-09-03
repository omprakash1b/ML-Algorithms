

import numpy as np
from collections import Counter


def euclidian_dis(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self,k=3):
        self.k = k

    def fit(self,X,y):
        self.X_train = X
        self.Y_train = y

    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self,x):
        #Compute the distance
        distances = [euclidian_dis(x,x_train) for x_train in self.X_train]

        # get closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_lables = [self.Y_train[i] for i in k_indices]

        #majority vote
        most_common = Counter(k_nearest_lables).most_common()
        return most_common[0][0]
    
    def eval(self,X,y):
        pred = self.predict(X)
        return round((np.sum(pred == y)/len(y))*100,2)