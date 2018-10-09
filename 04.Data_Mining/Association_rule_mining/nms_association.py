import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


alarm = pd.read_csv('/home/bhanuchander/alarm.csv')

## Data Organizing

names = ['OBJECTTYPE', 'LAYERRATE', 'NATIVEPROBABLECAUSE', 'SEVERITY']

alarm = alarm[names]

# datadict = pd.DataFrame(alarm.dtypes)
#
# datadict['MissingVal'] = alarm.isnull().sum()
#
# print datadict

for n in alarm.columns[alarm.isna().any()].tolist():

    unkwn = 'UNKNOWN_' + n

    alarm[n].fillna(unkwn, inplace=True)

    print '---------------------'

    print 'Updated Column Name : ', n

    print alarm[n].value_counts()

    print '\n-------------------'

# For Bigger Data
records = []
for i in range(0, alarm.shape[0]-1):
    records.append([str(alarm.values[i,j]) for j in range(0, alarm.shape[1])])

oht = TransactionEncoder()
oht_ary = oht.fit(records).transform(records)

print oht.columns_
df = pd.DataFrame(oht_ary, columns=oht.columns_)



frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
print (frequent_itemsets)

#-----------------------------------

rules = association_rules(frequent_itemsets, metric= "confidence", min_threshold=0.7)
print (rules)