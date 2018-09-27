from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()
logr = lmodel = make_pipeline(PolynomialFeatures(degree=2), Ridge())
boston = datasets.load_boston()
y = boston.target


# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:

predicted = cross_val_predict(lr, boston.data, y, cv=10)

polpredicted = cross_val_predict(logr, boston.data, y, cv = 10)

pred = [('linear',predicted), ('poly',polpredicted)]

plt.plot(y, label='actual', color='g')
plt.plot(predicted, label='linear predicted', color='r')
plt.plot(polpredicted, label='linear predicted', color='orange')

plt.show()