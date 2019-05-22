import pandas as pd
import numpy as np
from sklearn import preprocessing
#import statsmodels.api as sm

import matplotlib.pyplot as plt 
plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('./DataSets/trainEAR.csv',header = 0)
data = data.dropna()


dataTest = pd.read_csv('./DataSets/testEAR.csv',header = 0)
dataTest = dataTest.dropna()

print(dataTest['y'].value_counts())
#to plot 'y':

#sns.countplot(x='eyeDetection', data=data , palette='hls')
#plt.show()
#plt.savefig('count_plot')


#count_no_sub = len(data[data['eyeDetection']==0])
#count_sub = len(data[data['eyeDetection']==1])
#pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
#print("percentage of eyeblinks is", pct_of_no_sub*100)
#pct_of_sub = count_sub/(count_no_sub+count_sub)
#print("percentage of no blinks", pct_of_sub*100)

#matplotlib inline
#pd.crosstab(data.job,data.y).plot(kind='bar')
#plt.title('Purchase Frequency for Job Title')
#plt.xlabel('Job')
#plt.ylabel('Frequency of Purchase')
#plt.savefig('purchase_fre_job')
#plt.show()

data_vars=data.columns.values.tolist()
data_final = data[data_vars]

data_varsTest = dataTest.columns.values.tolist()
data_finalTest = dataTest[data_varsTest]

#data_final_vars=data.columns.values.tolist()
#y=['eyeDetection']
#X=[i for i in data_final_vars if i not in y]

X = data_final.iloc[:, 1:]
y = data_final.loc[:, data_final.columns == 'y']

X_test = data_finalTest.iloc[:, 1:]
y_test = data_finalTest.loc[:, data_finalTest.columns == 'y']

y_test = np.asarray(y_test)
#print(X_test)
#print(y)

logreg = LogisticRegression()
logreg.fit(X,y.values.ravel())

y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

y_pred = np.asarray(y_pred)
print(y_pred)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()



#filename = './Models/LogReg_modelEAR.sav'
#pickle.dump(logreg, open(filename, 'wb'))
