import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules



# dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
#            ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
#            ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
#            ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
#            ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]


titanic = pd.read_csv('/home/bhanuchander/course/Learn_MachineLearning/data/csv/titanic/train.csv')

nominal_cols = ['Embarked','Cabin','Pclass','Age', 'Survived']

in_titanic= titanic[nominal_cols]

in_titanic['Cabin'].fillna('Unknown',inplace=True)

in_titanic['Age'].fillna(in_titanic['Age'].mean(), inplace=True)

print in_titanic.isna().any()

# dataset = in_titanic.to_xarray()
#
# oht = TransactionEncoder()
# oht_ary = oht.fit(dataset).transform(dataset)
# df = pd.DataFrame(oht_ary, columns=oht.columns_)
# print (df)
#
# frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
# print (frequent_itemsets)
#
# association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
# rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
# print (rules)

#
# titanic = pd.read_csv('/home/bhanuchander/course/Learn_MachineLearning/data/csv/titanic/train.csv')
#
# nominal_cols = ['Embarked','Cabin','Pclass','Age', 'Survived']
#
# in_titanic= titanic[nominal_cols]
#
# print in_titanic.head()
#
# in_titanic['Cabin'].fillna('Unknown',inplace=True)
#
# in_titanic['Age'].fillna(in_titanic['Age'].mean(), inplace=True)
#
# print in_titanic.head()
#
# # Number of clusters
# k = 2
# # Number of training data
# n = in_titanic.shape[0]
#
# print n
# # Number of features in the data
# c = in_titanic.shape[1]
#
# data = in_titanic
#
# mean = np.mean(data, axis = 0)
# std = np.std(data, axis = 0)
#
# print std