import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

import warnings

warnings.filterwarnings('ignore')


train = pd.read_csv('/home/bhanuchander/course/Learn_MachineLearning/data/csv/titanic/train.csv')

train = train.set_index('PassengerId')

print train.shape

print train.Survived.value_counts(normalize=True)

train['Name_len']=train.Name.str.len()

train['Ticket_First']=train.Ticket.str[0]

train['FamilyCount']=train.SibSp+train.Parch

train['Cabin_First']=train.Cabin.str[0]

train['title'] = train.Name.str.extract('\, ([A-Z][^ ]*\.)',expand=False)

train.Fare.fillna(train.Fare.mean(),inplace=True)

trainML = train[['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
       'Fare', 'Embarked', 'Name_len', 'Ticket_First', 'FamilyCount',
       'title']]

nominal_cols = ['Embarked', 'Sex', 'Parch', 'Ticket_First', 'title']

trainML[nominal_cols] = trainML[nominal_cols].astype('category')

cat_columns = trainML.select_dtypes(['category']).columns

trainML[cat_columns] = trainML[cat_columns].apply(lambda x: x.cat.codes)

trainML.fillna(0, inplace=True)

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RF', RandomForestClassifier()))

print trainML.shape

X=trainML[['Age', 'SibSp', 'Parch',
       'Fare', 'Sex', 'Pclass','title', 'Name_len','Embarked', 'FamilyCount']] # Taking all the numerical values

Y = trainML['Survived'].values

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2, random_state=9)

for n, model in models:
    model.fit(X_train, Y_train)
    # Make a prediction
    y_predict = model.predict(X_validation)
    print " Model : ", n," Accuracy : ", (accuracy_score(Y_validation, y_predict))




train = pd.read_csv('/home/bhanuchander/course/Learn_MachineLearning/data/csv/titanic/test.csv')
train = train.set_index('PassengerId')

train['Name_len']=train.Name.str.len()

train['Ticket_First']=train.Ticket.str[0]

train['FamilyCount']=train.SibSp+train.Parch

train['Cabin_First']=train.Cabin.str[0]

train['title'] = train.Name.str.extract('\, ([A-Z][^ ]*\.)',expand=False)

train.Fare.fillna(train.Fare.mean(),inplace=True)

trainML = train[['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
       'Fare', 'Embarked', 'Name_len', 'Ticket_First', 'FamilyCount',
       'title']]

nominal_cols = ['Embarked', 'Sex', 'Parch', 'Ticket_First', 'title']

trainML[nominal_cols] = trainML[nominal_cols].astype('category')

cat_columns = trainML.select_dtypes(['category']).columns

trainML[cat_columns] = trainML[cat_columns].apply(lambda x: x.cat.codes)

trainML.fillna(0, inplace=True)

X_test=trainML[['Age', 'SibSp', 'Parch',
       'Fare', 'Sex', 'Pclass','title', 'Name_len','Embarked', 'FamilyCount']] # Taking all the numerical values


rf = RandomForestClassifier()

rf.fit(X, Y)

y_test = rf.predict(X_test)

trainML['Survived']=y_test



trainML.to_csv('out.csv')