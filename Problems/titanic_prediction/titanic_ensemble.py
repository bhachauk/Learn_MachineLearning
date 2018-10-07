import pandas as pd

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
from sklearn.cluster import KMeans
from sklearn.feature_selection import chi2, SelectKBest

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

titanic = pd.read_csv('/home/bhanuchander/course/Learn_MachineLearning/data/csv/titanic/train.csv')

nominal_cols = ['Embarked','Pclass', 'Sex', 'Parch', 'Cabin']

titanic['Embarked'].fillna('Unknown', inplace=True)

print titanic['Cabin'].value_counts()

con_titanic = titanic[['Age', 'SibSp', 'Fare', 'Pclass']]

## Binning Method to categorize the Continous Variables

def binning(col, cut_points, labels=None):

  minval = col.min()
  maxval = col.max()

  break_points = [minval] + cut_points + [maxval]


  if not labels:
    labels = range(len(cut_points)+1)


  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
  return colBin

cut_points = [1, 10, 20, 40, 60 ]

labels = ["Unknown", "Child", "Teen", "Adult", "Aged", "Old"]

con_titanic['Age'] = binning(con_titanic['Age'], cut_points, labels)

titanic['Embarked'].fillna('Unknown', inplace=True)

print con_titanic.head()



con_titanic[nominal_cols] = titanic[nominal_cols].astype('category')

cat_columns = con_titanic.select_dtypes(['category']).columns

con_titanic[cat_columns]= con_titanic[cat_columns].apply(lambda x: x.cat.codes)


print con_titanic.head()

Y = titanic['Survived']

X = con_titanic

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2, random_state=9)

seed = 9
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('KMean',KMeans(n_clusters=2)))
models.append(('chi', SelectKBest(chi2, k=2)))
# evaluate each model in turn
results = []
names=[]

for name, model in models:

    kfold = model_selection.KFold(n_splits=6, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)


    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

    model.fit(X_train, Y_train)
    pred = model.predict(X_validation)
    print accuracy_score(Y_validation, pred)




