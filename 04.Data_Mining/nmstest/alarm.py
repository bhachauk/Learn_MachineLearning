import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

import warnings
warnings.filterwarnings("ignore")


# Configuring plot fonts

font = {'size' : 6}

matplotlib.rc('font', **font)

alarm = pd.read_csv('/home/bhanuchander/alarm.csv')

alarm['TIME'] = pd.to_datetime(alarm['TIME'], unit='ms')

alarm['DAY'] = alarm['TIME'].dt.weekday_name

names = ['OBJECTTYPE', 'PROBABLECAUSE', 'SEVERITY', 'DAY']

alarm = alarm[names]

print 'Observations : ', alarm.shape[0]

print 'Columns      : ', alarm.shape[1]

# Always Fill na 'Unknown' for Nominal Data Set

# alarm['LAYERRATE'].fillna('LR_Unknown', inplace=True)

for n in names:

    print alarm[n].value_counts()



dataset = []
for i in range(0, alarm.shape[0]-1):
    dataset.append([str(alarm.values[i,j]) for j in range(0, alarm.shape[1])])
# dataset = alarm.to_xarray()

oht = TransactionEncoder()
oht_ary = oht.fit(dataset).transform(dataset)
df = pd.DataFrame(oht_ary, columns=oht.columns_)
print df.head()

print oht.columns_

output = apriori(df, min_support=0.1, use_colnames=oht.columns_)

print output

config = [
    ('antecedent support', 0.7),
    ('support', 0.3),
    # ('support', 0.3),
    ('confidence', 0.95),
    # ('conviction', 10)
]

for metric_type, th in config:
    rules = association_rules(output, metric=metric_type, min_threshold=th)
    if rules.empty:
        print 'Empty Data Frame For Metric Type : ',metric_type,' on Threshold : ',th
        continue
    print rules.columns.values
    print '----------------------------------------------'
    print 'For the Metric : ', metric_type, ' Value : ', th
    print '----------------------------------------------'
    print rules[['antecedents', 'consequents', metric_type]]

    support=rules.as_matrix(columns=['support'])
    confidence=rules.as_matrix(columns=['confidence'])

    plt.scatter(support, confidence, edgecolors='red')
    plt.xlabel('support')
    plt.ylabel('confidence')
    plt.title(metric_type+' : '+str(th))
    plt.show()
