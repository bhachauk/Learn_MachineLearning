from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn import model_selection


from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()
logr = lmodel = make_pipeline(PolynomialFeatures(degree=2), Ridge())
boston = datasets.load_boston()
y = boston.target


# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:

models = [('linear',lr), ('poly',logr)]



X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(boston.data, y, test_size=0.2, random_state=1)

for name, model in models:

    predicted = cross_val_predict(model, boston.data, y, cv=10)

    plt.plot(y, label='actual', color='g')
    plt.plot(predicted, label='linear predicted', color='r')
    plt.title(name)
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

    model.fit(X_train, Y_train)
    print 'Model Name : ', name, '   accuracy : ', model.score(X_validation, Y_validation)