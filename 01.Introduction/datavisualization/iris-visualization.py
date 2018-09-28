# Visualization of data

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import radviz

sns.set(style="white", color_codes=True)

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris = pd.read_csv("../../data/csv/iris.csv", names=names) # the iris dataset is now a Pandas DataFrame

# Head
print iris.head()

# Grouped Value
print iris["class"].value_counts()

# Pandas Plot

## Normal Plot with selected columns

iris.plot(kind="scatter", x="sepal-length", y="sepal-width")

plt.show()

iris.plot(kind="scatter", x="petal-length", y="petal-width")

plt.show()

diagkinds = ['auto', 'hist', 'kde']

for dk in diagkinds:

    sns.pairplot(iris, hue="class", height=3, diag_kind=dk)
    plt.show()

radviz(iris, "class")
plt.show()
