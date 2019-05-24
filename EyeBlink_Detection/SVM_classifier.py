# Required Packages
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import datasets		# To Get iris dataset
from sklearn import svm    			# To fit the svm classifier
import numpy as np
import matplotlib.pyplot as plt            # To visuvalizing the data
import pickle

# import iris data to model Svm classifier
#iris_dataset = datasets.load_iris()


#print("Iris data set Description :: ", iris_dataset['DESCR'])
#print("Iris feature data :: ", iris_dataset['data'])
#print("Iris target :: ", iris_dataset['target'])

#visuvalize_sepal_data()

#iris = datasets.load_iris()

data = pd.read_csv('./DataSets/trainEARBZ.csv',header = 0)
data = data.dropna()


dataTest = pd.read_csv('./DataSets/testEAR.csv',header = 0)
dataTest = dataTest.dropna()

print(dataTest['y'].value_counts())

data_vars=data.columns.values.tolist()
data_final = data[data_vars]

data_varsTest = dataTest.columns.values.tolist()
data_finalTest = dataTest[data_varsTest]

X = data_final.iloc[:, 1:]
y = data_final.loc[:, data_final.columns == 'y']
print(y.shape)
y = np.asarray(y)
y = y.flatten()

X = np.asarray(X)
print(X.shape)
X_test = data_finalTest.iloc[:, 1:]
y_test = data_finalTest.loc[:, data_finalTest.columns == 'y']

#X = iris.data[:, :2]  # we only take the Sepal two features.
#y = iris.target
C = 1.0  # SVM regularization parameter

## SVC with linear kernel
#svc = svm.SVC(kernel='linear', C=C).fit(X, y)
## LinearSVC (linear kernel)
#lin_svc = svm.LinearSVC(C=C).fit(X, y)
## SVC with RBF kernel
#rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
## SVC with polynomial (degree 3) kernel
#poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)

#X = iris.data[:, 2:]  # we only take the last two features.
#y = iris.target
#C = 1.0  # SVM regularization parameter
 
# SVC with linear kernel
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
# LinearSVC (linear kernel)
lin_svc = svm.LinearSVC(C=C).fit(X, y)
# SVC with RBF kernel
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
# SVC with polynomial (degree 3) kernel
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)


y_pred = rbf_svc.predict(X_test)
print(y_pred.shape)

print('Accuracy of SVM classifier on test set: {:.2f}'.format(rbf_svc.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#filename = './Models/SVM_modelBZ.sav'
#pickle.dump(rbf_svc, open(filename, 'wb'))
