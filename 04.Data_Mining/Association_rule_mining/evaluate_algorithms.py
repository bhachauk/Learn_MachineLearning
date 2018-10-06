import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pyfpgrowth

import warnings
warnings.filterwarnings("ignore")

import time

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

alarm = pd.read_csv('/home/bhanuchander/alarm.csv')

names = ['OBJECTTYPE', 'LAYERRATE', 'PROBABLECAUSEQUALIFIER', 'NATIVEPROBABLECAUSE', 'SEVERITY']

alarm = alarm[names]

alarm.fillna('Unknown', inplace=True)

alarm1 = pd.read_csv('/home/bhanuchander/cat.csv')

alarm1 = alarm1[names]

alarm1.fillna('Unknown', inplace=True)


#Data Org Finished
for in_titanic in [in_titanic, alarm, alarm1]:

    r = in_titanic.shape[0]
    c = in_titanic.shape[1]


    dataset = []
    ## Time For apriori
    start_time_ap = time.time()
    for i in range(0, r-1):
        dataset.append([str(in_titanic.values[i,j]) for j in range(0, c)])


    oht = TransactionEncoder()
    oht_ary = oht.fit(dataset).transform(dataset)
    df = pd.DataFrame(oht_ary, columns=oht.columns_)


    output = apriori(df, min_support=0.2, use_colnames=oht.columns_)
    rules = association_rules(output, metric='confidence', min_threshold=0.1)
    ap_time = (time.time() - start_time_ap)

    start_time = time.time()
    patterns = pyfpgrowth.find_frequent_patterns(in_titanic.values, 0.1)
    rules = pyfpgrowth.generate_association_rules(patterns, 0.1)
    fptime= (time.time() - start_time)


    print 'For Data Matrix          : ', r, ' x ', c
    print 'Number of Individuals    : ', df.shape[1]
    print 'Apriori                  : ', ap_time
    print 'FP-Algorithm             : ', fptime
    print '--------------------------'