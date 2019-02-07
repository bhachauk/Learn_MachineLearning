# some example data
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np

import pandas

import statsmodels.api as sm

from statsmodels.tsa.api import VAR, DynamicVAR

mdata = sm.datasets.macrodata.load_pandas().data

print mdata

# prepare the dates index

dates = mdata[['year', 'quarter']].astype(int).astype(str)
quarterly = dates["year"] + "Q" + dates["quarter"]

from statsmodels.tsa.base.datetools import dates_from_str

quarterly = dates_from_str(quarterly)

mdata = mdata[['realgdp','realcons','realinv']]

mdata.index = pandas.DatetimeIndex(quarterly)

data = np.log(mdata).diff().dropna()

# make a VAR model
model = VAR(data)

result = model.fit( 5)

print 'The Lag constant fitted is : ',result.k_ar
print result.summary()

result.plot()
plt.show()