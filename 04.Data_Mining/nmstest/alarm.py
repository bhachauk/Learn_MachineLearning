import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import chisquare

import numpy as np

# Configuring plot fonts

font = {'size' : 6}

matplotlib.rc('font', **font)

alarm = pd.read_csv('/home/bhanuchander/alarm.csv')

names = ['OBJECTTYPE', 'LAYERRATE', 'TYPE', 'FUNCTYPE', 'PROBABLECAUSEQUALIFIER', 'NATIVEPROBABLECAUSE', 'SEVERITY']

alarm = alarm[names]

print 'Observations : ', alarm.shape[0]

print 'Columns      : ', alarm.shape[1]

# Always Fill na 'Unknown' for Nominal Data Set

alarm.fillna('Unknown', inplace=True)

# for n in names:
#     grp = alarm[n].value_counts()
#
#     print grp
#     grp.plot(kind ='bar')
#     plt.title(n)
#     plt.gcf().subplots_adjust(bottom=0.15)
#     plt.show()

alarm = alarm.apply(lambda x : pd.factorize(x)[0])+1

print alarm.head()

print alarm.values

print pd.DataFrame([chisquare(alarm[x].values, f_exp=alarm.values.T, axis=1)[0] for x in alarm])

print '---------------------------'

# df = pd.DataFrame({'a':['a','b','a'],'b':['f','e','e']})
#
# df = df.apply(lambda x : pd.factorize(x)[0])+1
#
# print pd.DataFrame([chisquare(df[x].values,f_exp=df.values.T,axis=1)[0] for x in df])

