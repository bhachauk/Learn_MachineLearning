def get_feature_importances(clf, X_train, y_train=None,
                             top_n=10, figsize=(8, 8), enable_plot=True, print_table=False, title="Feature Importances"):


    __name__ = "get_feature_importances"

    import pandas as pd
    import matplotlib.pyplot as plt

    from xgboost.core import XGBoostError


    try:
        if not hasattr(clf, 'feature_importances_'):
            clf.fit(X_train.values, y_train.values.ravel())

            if not hasattr(clf, 'feature_importances_'):
                raise AttributeError("{} does not have feature_importances_ attribute".
                                     format(clf.__class__.__name__))

    except (XGBoostError, ValueError):
        clf.fit(X_train.values, y_train.values.ravel())

    feat_imp = pd.DataFrame({title : clf.feature_importances_})
    # feat_imp = pd.DataFrame(clf.feature_importances_, columns=[title])
    feat_imp['feature'] = X_train.columns
    feat_imp.sort_values(by= title, ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:top_n]

    feat_imp.sort_values(by= title, inplace=True, ascending=False)
    pltdf = feat_imp.set_index('feature', drop=True)

    if enable_plot:
        pltdf.plot.barh(title=title, figsize=figsize)
        plt.xlabel('Feature Importance Score')
        plt.show()

    if print_table:
        from IPython.display import display
        print("Top {} features in descending order of importance".format(top_n))
        display(pltdf.sort_values(by= title, ascending=False))

    return feat_imp, title

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
import matplotlib.pyplot as plt


clfs = [XGBClassifier(),
        ExtraTreesClassifier(),       ExtraTreeClassifier(),
        BaggingClassifier(),          DecisionTreeClassifier(),
        GradientBoostingClassifier(), LogisticRegression(),
        AdaBoostClassifier(),         RandomForestClassifier()]

url = "../data/csv/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

X_train = dataset.iloc[:,0:4]
Y_train = dataset['class']
result = pandas.DataFrame()
for clf in clfs:
    try:
        fi, classifier_name = get_feature_importances(clf, X_train, Y_train,enable_plot=False, top_n=X_train.shape[1], title=clf.__class__.__name__)
        if result.empty:
            result=fi
        else:
            result = pandas.merge(result, fi, on='feature')
    except AttributeError as e:
        print(e)

result = result.set_index('feature', drop=True)

print result
# TODO Get summary for best feature using by mean of all attributes