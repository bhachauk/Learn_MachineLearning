import pyfpgrowth
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

## Nominal Data Association With Apriori Algorithm

titanic = pd.read_csv('/home/bhanuchander/course/Learn_MachineLearning/data/csv/titanic/train.csv')


nominal_cols = ['Embarked','Pclass','Age', 'Survived', 'Sex']

in_titanic= titanic[nominal_cols]

in_titanic['Embarked'].fillna('Unknown',inplace=True)

in_titanic['Age'].fillna(0, inplace=True)

## Binning Method to categorize the Continous Variables

def binning(col, cut_points, labels=None):

  minval = col.min()
  maxval = col.max()

  break_points = [minval] + cut_points + [maxval]


  if not labels:
    labels = range(len(cut_points)+1)


  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
  return colBin

cut_points = [1, 10, 20, 50 ]

labels = ["Unknown", "Child", "Teen", "Adult", "Old"]

in_titanic['Age'] = binning(in_titanic['Age'], cut_points, labels)

# Replacing Binary with String
rep = {0: "False", 1: "True"}

in_titanic.replace({'Survived' : rep}, inplace=True)

print in_titanic.head()

patterns = pyfpgrowth.find_frequent_patterns(in_titanic.values, 5)

rules = pyfpgrowth.generate_association_rules(patterns, 0.1)

print in_titanic.shape[0]

Total_rows = float(in_titanic.shape[0])

for x,y in rules.iteritems():
  a_count = float(patterns[x])
  a_support = a_count / Total_rows
  print 'A : ', x, 'A_Count : ', a_count,' A_Support : ',a_support, '   Consequents : ', y