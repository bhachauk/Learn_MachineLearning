import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


alarm = pd.read_csv('/home/bhanuchander/alarm.csv')

names = ['OBJECTTYPE', 'LAYERRATE', 'PROBABLECAUSEQUALIFIER', 'NATIVEPROBABLECAUSE', 'SEVERITY']

alarm = alarm[names]

alarm.fillna('Unknown', inplace=True)

print alarm.info()

print alarm.shape[1]

records = []
for i in range(0, alarm.shape[0]-1):
    records.append([str(alarm.values[i,j]) for j in range(0, alarm.shape[1])])

oht = TransactionEncoder()
oht_ary = oht.fit(records).transform(records)

print oht.columns_
df = pd.DataFrame(oht_ary, columns=oht.columns_)



association_rules = apriori(df, min_support=0.3)


print association_rules



#
# frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
# print (frequent_itemsets)

#-----------------------------------

# association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
# rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
# print (rules)