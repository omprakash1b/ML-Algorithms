
'''
    prediction function

    y' = Xw + b

    error function (we use MSE here)
    j(w,b) = 1/n * Σ(y - y')^2

    optimization function (we use gradient descent here)
    w = w - α * ∂j(w,b)/∂w
    b = b - α * ∂j(w,b)/∂b

    α: learning rate

                 _          _           _                            _
                |∂j(w,b)/∂w  |         |    -2/n * Σ(y - y') * X       |                  
    j'(w,b) =   |∂j(w,b)/∂b  | =       |    -2/n * Σ(y - y')           |
                |_          _|         |_                            _|
'''
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Linear_Regression:
    def __init__(self,lr = 0.01,epoch = 1000):
        self.lr = lr
        self.epoch = epoch
        self.w = None
        self.b = None

    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epoch):
            y_pred = np.dot(X,self.w)+self.b
            dw = (1/n_samples)*np.dot(X.T,(y_pred-y))
            db = (1/n_samples)*np.sum(y_pred-y)

            self.w -= self.lr*dw
            self.b -= self.lr*db

    def predict(self,X):
        y_pred = np.dot(X,self.w)+self.b
        return y_pred
    
    def evaluation(self,X_test,y_test):
        y_pred = self.predict(X_test)
        mse = np.mean((y_test-y_pred)**2)
        count =0
        m = len(y_test)
        for i in range(m):
            if y_pred[i]!=y_test[i]:
                count+=1
        acc = count/m
        acc *= 100
        str = f"Accuracy = {acc}\nMSE = {mse}\ncount = {count}\nNo. of Samples = {m}"

        return str


if __name__ =="__main__":
    X,y = datasets.make_regression(n_samples = 1000,n_features =1,noise= 20 , random_state = 4)
    plt.scatter(X,y,color = 'b')
    plt.show()
    X_train,X_test,Y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state =24)
    reg = Linear_Regression()
    reg.fit(X_train,Y_train)
    p = reg.predict(X_test)
    print(p)
    eval = reg.evaluation(X_test,y_test)
    print(eval)




