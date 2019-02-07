## ARIMA
---

> **Auto Regressive Integrated Moving Average**

Very simple Uni-variate Time series prediction algorithm. Uses the moving average methodology
in time series data.

### Why we need VAR ?
---

> **Vector Auto Regression** 

When the time series data depends on other variables also, It means the future value is depends on
previous value and also current other variables. For example :


###### Uni-variate Data input :
---

|Time|Temp|
|----|----|
|t1|T1|
|t2|T2|
|.|.|
|tn|Tn|

###### Multi-variate Data input :
---

|Time|Temp|Humidity|Wind|
|----|----|--------|----|
|t1|T1|H1|W1|
|t2|T2|H2|W2|
|.|.|.|.|
|tn|Tn|Hn|Wn|