import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


class ChiSquare:

    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None  # P-Value
        self.chi2 = None  # Chi Test Statistic
        self.dof = None

        self.dfObserved = None
        self.dfExpected = None

    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        print(result)

    def TestIndependence(self, colX, colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)

        self.dfObserved = pd.crosstab(Y, X)
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof

        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index=self.dfObserved.index)

        self._print_chisquare_result(colX, alpha)



## Nominal Data Association With Apriori Algorithm

titanic = pd.read_csv('/home/bhanuchander/course/Learn_MachineLearning/data/csv/titanic/train.csv')


nominal_cols = ['Embarked','Pclass','Age', 'Survived', 'Sex']

in_titanic= titanic[nominal_cols]

in_titanic['Embarked'].fillna('Unknown',inplace=True)

in_titanic['Age'].fillna(0, inplace=True)

## Binning Method to categorize the Continous Variables

def binning(col, cut_points, labels=None):

  minval = col.min()
  maxval = col.max()

  break_points = [minval] + cut_points + [maxval]


  if not labels:
    labels = range(len(cut_points)+1)


  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
  return colBin

cut_points = [1, 10, 20, 50 ]

labels = ["Unknown", "Child", "Teen", "Adult", "Old"]

in_titanic['Age'] = binning(in_titanic['Age'], cut_points, labels)

# Replacing Binary with String
rep = {0: "False", 1: "True"}

in_titanic.replace({'Survived' : rep}, inplace=True)

print in_titanic.head()

# Data Org Finished

#Initialize ChiSquare Class
cT = ChiSquare(in_titanic)
#Feature Selection
nominal_cols.remove('Survived')
for var in nominal_cols:
    cT.TestIndependence(colX=var, colY="Survived")