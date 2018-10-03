import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




titanic = pd.read_csv('/home/bhanuchander/course/Learn_MachineLearning/data/csv/titanic/train.csv')

nominal_cols = ['Embarked','Cabin','Pclass','Age', 'Survived']

in_titanic= titanic[nominal_cols]

print in_titanic.head()

in_titanic['Cabin'].fillna('Unknown',inplace=True)

in_titanic['Age'].fillna(in_titanic['Age'].mean(), inplace=True)

print in_titanic.head()

# Number of clusters
k = 2
# Number of training data
n = in_titanic.shape[0]

print n
# Number of features in the data
c = in_titanic.shape[1]

data = in_titanic

mean = np.mean(data, axis = 0)
std = np.std(data, axis = 0)

print std
# centers = np.random.randn(k, c)*std + mean
#
# # Plot the data and the centers generated as random
# plt.scatter(data[:,0], data[:,1], s=7)
# plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)