from sklearn import datasets
from sklearn import model_selection, metrics
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np


from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()
logr = lmodel = make_pipeline(PolynomialFeatures(degree=1), Ridge())

a = [250, 300, 500, 800, 1000, 1250]

b = [2150, 2400, 3000, 4800, 7100, 8000]


area = np.array(a).reshape(-1,1)

cost = np.array(b).reshape((-1,1))

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:

predicted = cross_val_predict(lr, area, cost, cv=2)

polpredicted = cross_val_predict(logr, area, cost, cv=2)

models = [('linear',lr), ('poly',logr)]

plt.plot(area, cost, label='actual', color='g')
plt.plot(area, predicted, label='linear predicted', color='b')
plt.plot(area, polpredicted, label='linear predicted', color='r')

plt.show()


X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(area, cost, test_size=0.2, random_state=7)

print 'passed'

for name, model in models:
    kfold = model_selection.KFold(n_splits=2, random_state=1)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
    newpred = model_selection.cross_val_predict(model, X_train, Y_train)

    plt.plot(X_train, Y_train, label='actual', color='g')
    plt.plot(X_train, newpred, label='linear predicted', color='r')
    plt.title(name)
    plt.show()

    model.fit(X_train, Y_train)

    print 'Model Name : ',name, '   accuracy : ', model.score(X_validation, Y_validation)

    print cv_results.mean()