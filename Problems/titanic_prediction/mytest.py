import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#Get data
train = pd.read_csv('/home/bhanuchander/course/Learn_MachineLearning/data/csv/titanic/train.csv')
test = pd.read_csv('/home/bhanuchander/course/Learn_MachineLearning/data/csv/titanic/test.csv')


train.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)
test.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)
train.head()

one_hot_train = pd.get_dummies(train)
one_hot_test = pd.get_dummies(test)

# First five rows from train dataset
one_hot_train.head()

one_hot_train['Age'].fillna(one_hot_train['Age'].mean(), inplace=True)
one_hot_test['Age'].fillna(one_hot_test['Age'].mean(), inplace=True)
one_hot_train.isnull().sum()

one_hot_test.isnull().sum().sort_values(ascending=False)

# Fill the null Fare values with the mean of all Fares
one_hot_test['Fare'].fillna(one_hot_test['Fare'].mean(), inplace=True)
one_hot_test.isnull().sum().sort_values(ascending=False)

feature = one_hot_train.drop('Survived', axis=1)
target = one_hot_train['Survived']

# Model creation
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=1, criterion='gini', max_depth=10, n_estimators=50, n_jobs=-1)
rf.fit(feature, target)