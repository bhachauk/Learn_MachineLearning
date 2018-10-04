import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

## Nominal Data set visualization

titanic = pd.read_csv('/home/bhanuchander/course/Learn_MachineLearning/data/csv/titanic/train.csv')

nominal_cols = ['Embarked','Cabin','Pclass','Age', 'Survived']

in_titanic= titanic[nominal_cols]

print in_titanic.head()

in_titanic['Cabin'].fillna('Unknown',inplace=True)

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

print in_titanic['Age'].value_counts()

print in_titanic.describe()


# Number of training data
n = in_titanic.shape[0]

# Number of features in the data
c = in_titanic.shape[1]


print 'Data Set Rows  : ',n, '  Columns  : ',c



## Cross Tabulation.
for n in nominal_cols:
    for m in nominal_cols:
        if n == m or len(in_titanic[n].value_counts()) > 10 or len(in_titanic[m].value_counts())> 10:
            continue
        out = pd.crosstab(in_titanic[n], in_titanic[m])
        print out
        out.plot.bar(stacked=True)
        plt.show()
        # Plotting In Sea Born

        ax = plt.subplot()
        sns.heatmap(out, annot=True, ax=ax)

        # labels, title and ticks
        ax.set_xlabel(m)
        ax.set_ylabel(n)
        title = 'Heat Map : '+ n+ ' Vs '+ m
        ax.set_title(title)
        plt.show()

## visualization Area Completed.....