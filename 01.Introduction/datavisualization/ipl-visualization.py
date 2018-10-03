import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

ipl = pd.read_csv("../../data/csv/ipl.csv") # the iris dataset is now a Pandas DataFrame

print ipl.info()

for cor in ['pearson', 'kendall', 'spearman']:

    correlations = ipl.corr(method=cor)

    print correlations.head()
    # plot correlation matrix
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(correlations)
    # fig.colorbar(cax, cmap='RdBu')
    # ax.set_xticklabels(names)
    # ax.set_yticklabels(names)
    # plt.show()