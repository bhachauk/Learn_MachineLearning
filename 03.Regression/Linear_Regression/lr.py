import numpy as np
import warnings
from matplotlib import pyplot as plt
import random

from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.neural_network import MLPRegressor
import keras

warnings.filterwarnings("ignore")


def simple_model():
    return keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])


# Case -1
a = range(1000)
b = [(2*x-2) for x in a]

# Case - 2
# a = range(20)
# a = [random.randrange(50) for _ in a]
# b = [x+1 for x in a]


# # Case - 3
# a = range(50)
# a = [random.randrange(100) for _ in a]
# b = [x+1 if x % 2 == 0 else x-1 for x in a]

input_data = np.array(a).reshape(-1, 1)
target_data = np.array(b).reshape((-1, 1))

# print("Input:")
# for i, x in enumerate(input_data):
#     print(input_data[i], target_data[i])

# plt.plot(a, b)

lr = linear_model.LinearRegression()
logr = make_pipeline(PolynomialFeatures(degree=3), Ridge())
tf = simple_model()
tf.compile(optimizer='sgd', loss='mean_squared_error')
mlp = MLPRegressor(hidden_layer_sizes=(3), activation='tanh', solver='lbfgs')


models = [(lr, 'linear'), (logr, 'poly'), (mlp, 'mlp'), (tf, 'tf')]

# organizing
for model, name in models:
    model.fit(input_data, target_data) if not name == 'tf' else model.fit(input_data, target_data, epochs=1000, verbose=False)
    predicted = list()
    for i in a:
        val = model.predict([[i]])[0]
        val = val[0] if name is not 'mlp' else val
        predicted.append(val)
    plt.plot(a, predicted)
    print('Model: {} predicted : {}'.format(name, model.predict([[20]])))
plt.show()
# 1 - Recap (Overview)
# 2 - Hands_on_models (Models related intro - Example : sk_learn)
# 3 - Numbering system learning. (Linear, random)
# 4 - Non-linearity models (Milk - problem, fitting category, accuracy : rmse
