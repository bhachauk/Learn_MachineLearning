# Linear Regression with Iris Data set:
# Ref : https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
import pandas
import numpy as np
from pandas import Series
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import seaborn as sns



# Load dataset
url = "/home/bhanuchander/course/Learn_MachineLearning/data/csv/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

print dataset.describe()

print dataset.groupby('class').size()


scatter_matrix(dataset, c=['r', 'g', 'b'])
plt.show()


array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20

temp = dataset
narray = temp.values
tX = narray[:,0:4]
tY = narray[:,4]
mylabel = np.unique(tY)

tY[tY == 'Iris-setosa'] = 1
tY[tY == 'Iris-versicolor'] = 2
tY[tY == 'Iris-virginica'] = 3
for x in range(4):
	plt.scatter(tY, tX[:, x])
	plt.xticks([1, 2, 3], mylabel)
	plt.title(names[x])
	plt.show()


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

	predicted = model_selection.cross_val_predict(model, X_train, Y_train)
	cm= confusion_matrix(Y_train, predicted)
	ax = plt.subplot()
	sns.heatmap(cm, annot=True, ax=ax)
	#annot=True to annotate cells

	# labels, title and ticks
	ax.set_xlabel('Predicted labels')
	ax.set_ylabel('True labels')
	ax.set_title('Confusion Matrix')
	plt.show()

	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


fig= plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
