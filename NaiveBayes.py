import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

def accuracy(ypred,y):
    accuracy = np.sum(ypred == y)/len(y)
    return accuracy

class NaiveBayes:
    
    def fit(self, X,y):
        n_samples,n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        #calculation of mean, variance and prior for each class
        self._mean = np.zeros((n_classes,n_features),dtype = np.float64)
        self._var = np.zeros((n_classes,n_features),dtype = np.float64)
        self._prior = np.zeros(n_classes,dtype = np.float64)

        for idx,clas in enumerate(self._classes):
            X_c = X[y==clas]
            self._mean[idx,:] = X_c.mean(axis = 0)
            self._var[idx,:] = X_c.var(axis = 0)
            self._prior[idx] = X_c.shape[0]/float(n_samples)

    def _predict(self,x):
        posteriors = []

        #calculation of posterior probability for each class
        for idx,c in enumerate(self._classes):
            prior = np.log(self._prior[idx])
            posterior = np.sum(np.log(self._pdf(idx,x)))
            posterior += prior
            posteriors.append(posterior)

            #return the class with heighest posteriors
        return self._classes[np.argmax(posteriors)]
        
    def _pdf(self,class_idx,x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x-mean)**2)/(2*var))
        denominator = np.sqrt(2*np.pi*var)

        return numerator/denominator



    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    


if __name__ == "__main__":
    X,y = datasets.make_classification(
        n_samples = 10000,n_features = 10,n_classes =2,random_state = 123
    )

    X_train,X_test,Y_train,y_test = train_test_split(
        X,y,random_state = 24 ,test_size = 0.25
    )

    nb = NaiveBayes()
    nb.fit(X_train,Y_train)
    predictions = nb.predict(X_test)

    print(predictions)
    print(f"\nAccuracy {accuracy(predictions,y_test)*100}%")