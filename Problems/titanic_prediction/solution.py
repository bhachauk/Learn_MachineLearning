import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

titanic = pd.read_csv('/home/bhanuchander/course/Learn_MachineLearning/data/csv/titanic/train.csv')

def getOrg(in_train ):


    in_train['Name_len']=in_train.Name.str.len()

    in_train['Ticket_First']=in_train.Ticket.str[0]

    in_train['FamilyCount']= in_train.SibSp + in_train.Parch

    in_train['Cabin_First']=in_train.Cabin.str[0]

    in_train.drop(columns=['Cabin'])

    # Regular expression to get the title of the Name
    in_train['title'] = in_train.Name.str.extract('\, ([A-Z][^ ]*\.)', expand=False)

    in_train.title.value_counts().reset_index()

    in_train.Fare.fillna(in_train.Fare.mean(), inplace=True)

    in_train.Fare.replace(0, in_train.Fare.mean(), inplace=True)

    # in_train = pd.get_dummies(in_train)

    nominal_cols = ['Embarked', 'Pclass', 'Sex', 'Parch', 'Cabin_First','Ticket_First',  'title']

    in_train[nominal_cols] = in_train[nominal_cols].astype('category')

    cat_columns = in_train.select_dtypes(['category']).columns

    in_train[cat_columns] = in_train[cat_columns].apply(lambda x: x.cat.codes)

    # in_train = in_train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
    #    'Fare', 'Embarked', 'Name_len', 'Ticket_First', 'FamilyCount', 'title']]

    in_train = in_train[['Survived','Age', 'SibSp', 'Parch',
       'Fare', 'Name_len', 'FamilyCount']]

    in_train = in_train.dropna()

    return in_train

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RF', RandomForestClassifier()))

train = getOrg(titanic)

print train.head()

Y = train['Survived']

filtercol = list(train)

filtercol.remove('Survived')

X = train[filtercol]

print X.info()


X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2, random_state=9)

seed = 9
scoring = 'accuracy'

# evaluate each model in turn
results = []
names=[]

for name, model in models:

    kfold = model_selection.KFold(n_splits=6, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

    model.fit(X_train, Y_train)
    pred = model.predict(X_validation)
    print accuracy_score(Y_validation, pred)

#
# # cart = DecisionTreeClassifier()
# #
# # cart.fit(X, Y)
# #
# # titanic = pd.read_csv('/home/bhanuchander/course/Learn_MachineLearning/data/csv/titanic/test.csv')
# #
# # train = getOrg(titanic)
# #
# #
# # X = train
# #
# # pred = cart.predict(X)
