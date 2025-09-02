'''
    prediction function
    y_pred = h-theta(x) = 1/(1+exp(-wx+b))

    error function
    j(w,b) = j(theta) = 1/n *summation(y[i]log(h-theta(x[i]))+(1-y[i])log(1-h-theta(x[i])))

    
                 _          _           _                            _
                |∂j(w,b)/∂w  |         |    -2/n * Σ(y - y') * X       |                  
    j'(w,b) =   |∂j(w,b)/∂b  | =       |    -2/n * Σ(y - y')           |
                |_          _|         |_                            _|

'''
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Logistic_Regression:
    def __init__(self, lr=0.01,epoch = 1000):
        self.w = None
        self.b = None
        self.lr = lr
        self.epoch = epoch

    def sigmoid(self,fx):
        fx = np.clip(fx, -500, 500)
        return 1/(1+np.exp(-fx))

    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        

        for _ in range(self.epoch):

            dotp = np.dot(X,self.w) +self.b
            y_pred = self.sigmoid(dotp)

            dw = (1/n_samples)*np.dot(X.T,(y_pred-y))
            db = (1/n_samples)*np.sum(y_pred-y)

            self.w -= self.lr*dw
            self.b -= self.lr*db

    def predict(self,X):
        dotp = np.dot(X,self.w) +self.b
        y_pred = self.sigmoid(dotp)

        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred
    
    def eval(self,X,y):
        y_pred = self.predict(X)
        return np.sum(y==y_pred)/len(y)
    

if __name__ =="__main__":
    bc = datasets.load_breast_cancer()
    X,y = bc.data,bc.target
    
    X_train,X_test,Y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state =24)
    reg = Logistic_Regression()
    reg.fit(X_train,Y_train)
    p = reg.predict(X_test)
    print(p)
    print(f'Accuracy = {reg.eval(X_test,y_test)*100}%')
    
