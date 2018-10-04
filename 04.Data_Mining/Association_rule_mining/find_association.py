import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

import warnings
warnings.filterwarnings("ignore")
import seaborn as sns

## Nominal Data Association With Apriori Algorithm

titanic = pd.read_csv('/home/bhanuchander/course/Learn_MachineLearning/data/csv/titanic/train.csv')

nominal_cols = ['Embarked','Pclass','Age', 'Survived']

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

# Data Org Finished

dataset = []
for i in range(0, in_titanic.shape[0]-1):
    dataset.append([str(in_titanic.values[i,j]) for j in range(0, in_titanic.shape[1])])
# dataset = in_titanic.to_xarray()

oht = TransactionEncoder()
oht_ary = oht.fit(dataset).transform(dataset)
df = pd.DataFrame(oht_ary, columns=oht.columns_)
print df.head()

print oht.columns_

output = apriori(df, min_support=0.2, use_colnames=oht.columns_)

print output.head()

for x in ['Embarked', 'Pclass','Age']:
    sns.set(style="whitegrid")
    ax = sns.countplot(y=x, hue="Survived", data=in_titanic)
    plt.ylabel(x)
    plt.title('Survival Plot')
    plt.show()

config = [
    ('antecedent support', 0.7),
    ('support', 0.5),
    ('support', 0.3),
    ('confident', 0.5)
]

for metric_type, th in config:
    rules = association_rules(output, metric=metric_type, min_threshold=th)
    if rules.empty:
        print 'Empty Data Frame For Metric Type : ',metric_type,' on Threshold : ',th
        continue
    print rules.columns.values
    print (rules)

    support=rules.as_matrix(columns=['support'])
    confidence=rules.as_matrix(columns=['confidence'])

    plt.scatter(support, confidence, edgecolors='red')
    plt.xlabel('support')
    plt.ylabel('confidence')
    plt.title(metric_type+' : '+str(th))
    plt.show()
