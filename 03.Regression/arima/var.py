# Implementing VAR
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#read the data
df = pd.read_csv("/home/bhanuchander/data/AirQualityUCI/AirQualityUCI.csv", parse_dates=['Date', 'Time'], sep=';')

#check the dtypes
print df.info()

# df['Date_Time'] = df['Date'].map(str) + df['Time'].map(str)
#
# df['Date_Time'] = pd.to_datetime(df.Date_Time , format = '%d/%m/%Y %H.%M.%S')
data = df.drop(['Date', 'Time'], axis=1)

#missing value treatment
cols = data.columns
for j in cols:
    for i in range(0,len(data)):
       if data[j][i] == -200:
           data[j][i] = data[j][i-1]
           print 'i : {}, j : {}'.format(i, j)

#checking stationarity
from statsmodels.tsa.vector_ar.vecm import coint_johansen
#since the test works for only 12 variables, I have randomly dropped
#in the next iteration, I would drop another and check the eigenvalues

johan_test_temp = data.drop(['CO(GT)'], axis=1)

eigen = coint_johansen(johan_test_temp, -1, 1).eig

#creating the train and validation set
train = data[:int(0.8*(len(data)))]
valid = data[int(0.8*(len(data))):]

#fit the model
from statsmodels.tsa.vector_ar.var_model import VAR
from math import sqrt

model = VAR(endog=train)
model_fit = model.fit()

# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(valid))

pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
for j in range(0,13):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j]

#check rmse
for i in cols:
    print('rmse value for', i, 'is : ', sqrt(mean_squared_error(pred[i], valid[i])))