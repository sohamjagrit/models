import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split

from knn import KNN

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(X.shape, y.shape)
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)
# print(X_train[0], y_train[0])
# print(X_test[0], y_test[0])
# print(y_train)

clf = KNN(k=3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"Accuracy: {accuracy}")
