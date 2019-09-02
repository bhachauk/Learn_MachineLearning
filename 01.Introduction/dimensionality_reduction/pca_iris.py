import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True)
di = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3 }

df['class'].replace(di, inplace=True)

feat_cols = list(df)[:4]
print ('Feature columns : {}'.format(','.join(feat_cols)))
X = df.ix[:, 0:4].values
y = df.ix[:, 4].values


pca = PCA(n_components=4)
pca_result = pca.fit_transform(df[feat_cols].values)

pca_df = pd.DataFrame(pca_result, columns=['pc1', 'pc2', 'pc3', 'pc4'])

print ('PCA Explained Variance Ratio : ', pca.explained_variance_ratio)
plt.bar(range(4), pca.explained_variance_ratio_)
plt.show()


models = list()
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

print (np.array(pca_df['pc1']).reshape(-1, 1))

for name, model in models:
    cross_score = cross_val_score(model, np.array(pca_df['pc1']).reshape(-1, 1), df['class'], cv=3, verbose=0)
    print ('Model : {}, Score : {}'.format(name, np.mean(cross_score)))

# predicted = cross_val_predict(lr, pca_df.values, df['class'], cv=3, verbose=1)
# score= accuracy_score(df['class'], predicted)
# print score
# plt.plot(y, label='actual', color='g')
# plt.plot(predicted, label='linear predicted', color='r')
# plt.show()


