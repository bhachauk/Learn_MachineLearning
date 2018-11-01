import pandas as pd
import numpy as np
import sys
sys.setrecursionlimit(20000)

import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('/home/bhanuchander/course/Learn_MachineLearning/data/csv/titanic/train.csv')


def getFormat (train):



    train['Ticket'].fillna(0, inplace=True)

    # train['isSpecial'] = train.Name.str.contains('(', regex=False)

    train['Ticket_Type']=train.Ticket.apply(lambda x : x.isalnum())

    train['Ticket']= train.Ticket.str.extract('(\d+)')

    train['title'] = train.Name.str.extract('\, ([A-Z][^ ]*\.)',expand=False)

    train.Fare.fillna(0, inplace=True)

    train.Age.fillna(0, inplace=True)

    train['Cabin_First'] = train.Cabin.str[0]

    train['Cabin_Number'] = train.Cabin.str.replace('[^0-9]', '')

    train['Cabin'].fillna('UNKNOWN-CABIN', inplace=True)

    train['Cabin'] = train.Cabin.str.replace('[^a-zA-Z]', '')

    train['Cabin_Unique'] = train['Cabin'].apply( lambda x : len(set((str(x)))))

    train['Cabins'] = train['Cabin'].apply( lambda x : ''.join(set((str(x)))))

    train['Cabin'] = train['Cabin'].apply (lambda x : len(str(x)))

    trainML = train[['Pclass', 'Sex', 'SibSp', 'Parch','PassengerId',
                     'Fare', 'Embarked', 'Ticket_Type', 'title',
                     'Cabin', 'Cabin_First', 'Cabin_Unique', 'Ticket',
                     'Cabins', 'Cabin_Number']]

    trainML.fillna('Unknown', inplace=True)

    nominal_cols = ['Sex','SibSp', 'Pclass', 'title', 'Embarked', 'Cabin_First', 'Cabins']

    # X = pd.get_dummies(trainML[nominal_cols])

    X = pd.DataFrame()

    trainML[nominal_cols] = trainML[nominal_cols].astype('category')

    cat_columns = trainML.select_dtypes(['category']).columns

    X[cat_columns] = trainML[cat_columns].apply(lambda x: x.cat.codes)

    return X


X = getFormat(train=train)

Y = train['Survived']

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.25, random_state = 5)

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RF', RandomForestClassifier()))

vc = VotingClassifier(estimators = models, voting='soft', weights=[2, 1, 2, 1, 2, 1, 2, 1])
models.append(('Voting', vc))
for n, model in models:
    model.fit(X_train, Y_train)
    # Make a prediction
    y_predict = model.predict(X_validation)
    print " Model : ", n," Accuracy : ", (accuracy_score(Y_validation, y_predict))

# Generating Output :

train = pd.read_csv('/home/bhanuchander/course/Learn_MachineLearning/data/csv/titanic/test.csv')

X_test = getFormat(train)

prime = list(X)

rf = RandomForestClassifier(n_estimators=100)

rf.fit(X, Y)

y_predict = rf.predict(X_test)

train['Survived'] = y_predict

train[['Survived','PassengerId']].to_csv('out.csv', index=False)