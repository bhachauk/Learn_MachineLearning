### Data Mining
---

> Converting Data Into Interesting Informations

> *“Data mining is the process of exploration and analysis, by automatic or semiautomatic means, of large quantities of data in order to discover meaningful patterns and rules.”*
(M. J. A. Berry and G. S. Linoff)"

> *“Data mining is finding interesting structure (patterns, statistical models, relationships) in databases.”* (U. Fayyad, S. Chaudhuri and P. Bradley)


Data mining is Multi-disciplinary.

- Data Processing / Manipulation
- Visualization
- Machine Learning and AI


#### How it related ?
---

<p align="center">
<i>Relation Between The Terminologies</i>
</p>
<p align="center">
<kbd>
<img src="/data/img/data-science.jpg" width="500" height="500"/></kbd> 
</p>

Img Source : [data-science-central](https://www.datasciencecentral.com/profiles/blogs/difference-of-data-science-machine-learning-and-data-mining)


#### Some Relational Difference Between DM - ML
---

<p align="center">
<i>Diff Between Data Mining and Machine Learning</i>
</p>
<p align="center">
<kbd>
<img src="/data/img/dm-vs-ml.jpg" width="800" height="2000"/></kbd> 
</p>


Img Source : [educba](https://www.educba.com/data-mining-vs-machine-learning/)


### Goal
---

*Turning Data into Interesting Information*

##### Techniques:
---

- Classification 

Examining the feature of a newly presented object and assigning it to one of a predefined set of classes;

- Estimation 

Given some input data, coming up with a value for some unknown continuous variable such as income, height, or credit-card balance;

- Prediction 

The same as classification and estimation except that the records are classified according to some predicted future behaviour or estimated future value;

- Affinity Grouping or Association Rules 

Determine which things go together, also known as dependency modelling, E.g. In a shopping cart at the supermarket – market basket analysis;

- Clustering 

Segmenting a population into a number of subgroups or clusters and Description and visualization exploratory or visual data mining.

- Feature Analysis

Analysing the input feature strength.


#### Input Data Types:
---
 
*Larger Amount Of Dataset*

- Continuous
- Nominal / Categorical

###### Continuous Data:
---

Ex : Stock Rate, Marks, etc… 

- Pearson Correlation.
- Spearman rank correlation.
- Kendall correlation.

###### Categorical Data:
---

Ex: Gender, Class, Complexion, Blood Group, etc.. 

- Apriori Algorithm, FP Growth Algorithm.
- Chi Square. (For Prediction and Feature Learning)(Type of Correlation)

---


## How to get Interesting statistics from a Data ?
---

- Feature Analysis
- Association Rule Mining

---

#### Feature Analysis 

Analysing and evaluating which feature is important for predicting the target. 
Also used in Dimensionality Reduction. It Means It focuses On One Parameter to evaluate its Strength on Controlling Target.

Example : India Won All (100 %) Matches with Rohit Sharma’s Captaincy (It is not real ... :P )  

Data Set:
	
|Is Rohit the Captain?|Win?|
|---------------------|----|
|True|True|
|False|False|
|True|True|
|True|True|

> Here It is More predictable in first sight.. But What we do if we have **N** numbers of attributes...

Example Feature Analysis On :

- [IRIS Flower Data Set](/04.Data_Mining/Feature-Learning/feature_importance.ipynb)
- [Titanic Data Set](/04.Data_Mining/Feature-Learning/Chi2_Test_Featue_Importance_For_Titanic_Data.ipynb)

#### Awesome Posts
---

- [http://tdan.com/data-mining-and-statistics-what-is-the-connection/5226#](http://tdan.com/data-mining-and-statistics-what-is-the-connection/5226#)
- [https://www.linkedin.com/pulse/3-styles-data-mining-abhishek-mittal/](https://www.linkedin.com/pulse/3-styles-data-mining-abhishek-mittal/)

- [Ipl Data - paper](http://www.ijmlc.org/papers/143-C00120-003.pdf)
- [Machine Learning On Feature Extraction - paper](https://arxiv.org/pdf/1711.10933.pdf)


- [Data Mining for Fun and Profit](https://www.jstor.org/stable/2676725?seq=1#page_scan_tab_contents)
- [Discussion On Correlation](https://www.researchgate.net/post/Correlation_between_discrete_and_categorical_data)

- [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/)
- [Machine Learning Mastery](https://machinelearningmastery.com/feature-selection-machine-learning-python/)
- [Correlation on Signals](https://www.allaboutcircuits.com/technical-articles/understanding-correlation/)
- [Correlation Types](http://www.statisticssolutions.com/correlation-pearson-kendall-spearman/)
- [Feature Performance](https://www.kaggle.com/grfiv4/plotting-feature-importances)
- [Feature importance on Titanic for Chi 2](http://www.handsonmachinelearning.com/blog/2AeuRL/chi-square-feature-selection-in-python)

- [Data Types](https://statistics.laerd.com/statistical-guides/types-of-variable.php)
- [concordant and discordant](https://www.statisticshowto.datasciencecentral.com/concordant-pairs-discordant-pairs/)

- [visualizing nominal set](http://adataanalyst.com/data-analysis-resources/visualise-categorical-variables-in-python/)
- [chi 2 on multiple categorical values](https://stackoverflow.com/questions/48035381/correlation-among-multiple-categorical-variables-pandas)

#### Association Rule Mining
---

*Apriori Algorithm*

- [Example 1](http://pbpython.com/market-basket-analysis.html)
- [Example 2](https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/)
- [Example 3](https://www.analyticsvidhya.com/blog/2017/08/mining-frequent-items-using-apriori-algorithm/)
- [Example 4](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)
- [Efficient - Apriori - Python 3.6](https://pypi.org/project/efficient-apriori/)
- [Example 5 With Plots](http://intelligentonlinetools.com/blog/tag/apriori-algorithm-in-data-mining/)

*FP Growth*

- [FP Growth](https://pypi.org/project/pyfpgrowth/)

*Chi2*

- [Example 1](https://www.spss-tutorials.com/chi-square-independence-test/)
- [Example 2](http://statisticsbyjim.com/hypothesis-testing/chi-square-test-independence-example/)

#### Feature Learning
---
 
- [Feature Learning With Chi 2](https://www.linkedin.com/pulse/chi-square-feature-selection-python-md-badiuzzaman-biplob/)



**[BACK](/README.md)**