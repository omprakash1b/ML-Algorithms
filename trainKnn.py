import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Knn import KNN
from matplotlib.colors import ListedColormap


if __name__ == "__main__":
    iris = datasets.load_iris()
    X,y = iris.data,iris.target
    cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

    X_train,x_test,Y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 24)

    clf = KNN(k = 5)
    clf.fit(X_train,Y_train)
    y_pred = clf.predict(x_test)
    print(y_pred)
    print(
        f'\nAccuracy = {clf.eval(x_test,y_test)}'
    )



'''plt.figure
    plt.scatter(X[:,2],X[:,3],edgecolor = 'k',s = 20,cmap = cmap,c = y)
    plt.show()'''
    