#-------------

from xgboost              import XGBClassifier
from sklearn.ensemble     import ExtraTreesClassifier
from sklearn.tree         import ExtraTreeClassifier
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import GradientBoostingClassifier
from sklearn.ensemble     import BaggingClassifier
from sklearn.ensemble     import AdaBoostClassifier
from sklearn.ensemble     import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas
from sklearn.feature_selection import RFE
import warnings
warnings.filterwarnings('ignore')


clfs = [XGBClassifier(),
        ExtraTreesClassifier(),
        ExtraTreeClassifier(),
        BaggingClassifier(),
        DecisionTreeClassifier(),
        GradientBoostingClassifier(),
        LogisticRegression(),
        AdaBoostClassifier(),
        RandomForestClassifier()]

url = "../../data/csv/iris.csv"

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

dataset = pandas.read_csv(url, names=names)

X_train = dataset.iloc[:, 0:4]
Y_train = dataset['class']
result = pandas.DataFrame()
for clf in clfs:
    cname = clf.__class__.__name__
    print 'Classifier Name : ', cname
    try:
        rfe = RFE(clf)
        fit = rfe.fit(X_train, Y_train)
        print ("Num Features: %d") % fit.n_features_
        print ("Selected Features: %s") % [d for (d, remove) in zip(names,fit.support_) if not remove]
        print ("Feature Ranking: %s") % fit.ranking_

    except:
        print "Can't run."