# Contents Of Basics
---

<p align="center">
<b><i>Learn First Do Next</i></b>
</p>

---



Here I have organized most of the basics in chapter wise. I hope it will help you to explore fast. 

- [01. Introduction](01.Introduction/readme.md)
- [02. Classification](02.Classification/README.md)
- [03. Regression](03.Regression/readme.md)
- [04. DataMining](04.Data_Mining/README.md)


# Machine Learning - A Quick Note
---

> "the field of study that gives computers the ability to learn without being explicitly programmed" by Arthur Samuel.

> "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E " by Tom Mitchell.

Types:

- [Supervised](#supervised-learning)
- [Unsupervised](#unsupervised-learning)
- [Reinforcement](#-reinforcement-learning)

---

## Supervised Learning

*With Data Sets or Observations*

Types:

- [Classification](#-classification)
- [Regression](#-regression)
- [Ensembling](#)

> *Algorithms :* 
> Linear Regression, Logistic Regression, CART, Naive Bayes, KNN

#### Classification
---

In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.

Ex : 

- Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.
- Digit Recognition which produces output [0 - 9].


#### Regression
---

In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function.

Ex:

- Given a picture of a person, we have to predict their age on the basis of the given picture.
- Finding the flat price by space.

#### Ensembling
---

It is a type of supervised learning. It means combining the predictions of multiple different weak ML models to predict on a new sample. 

Ex:

Bagging with Random Forests, Boosting with XGBoost are examples of ensemble techniques.


## Unsupervised Learning
---

> Without Data Sets or Observations

We don't have any idea what our results should look like and 
there is no feedback based on the prediction results.

Types:

- [Clustering](#-clustering)
- [Non-Clustering](#-non-clustering)
- [Association](#-association)
- [Dimensionality Reduction](#-dimensionality-reduction)

Algorithms :

*Apriori, K-means, PCA*


#### Clustering
---

To group samples such that objects within the same cluster are more similar to each other than to the objects from another cluster.

Ex:

- Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

#### Non-Clustering
---

The Cocktail Party Algorithm, allows you to find structure in a chaotic environment. 

Example:

- Identifying individual voices and music from a mesh of sounds at a cocktail party.

#### Association
---

To discover the probability of the co-occurrence of items in a collection. 
It is extensively used in market-basket analysis. 

Example: 

- If a customer purchases bread, he is 80% likely to also purchase eggs.

#### Dimensionality Reduction
---

True to its name, Dimensionality Reduction means reducing the number of variables of a data set while ensuring that important information is still conveyed. 
Dimensionality Reduction can be done using Feature Extraction methods and Feature Selection methods. Feature Selection selects a subset of the original variables. 
Feature Extraction performs data transformation from a high-dimensional space to a low-dimensional space. 

Example: 

- PCA algorithm (Principal Component Analysis) is a Feature Extraction approach.

    
## Reinforcement learning
---
 
 *Not supervised nor unsupervised*

Reinforcement learning is a type of machine learning algorithm that allows the agent to decide the best next action based on its current state, 
by learning behaviours that will maximize the reward. Reinforcement algorithms usually learn optimal actions through trial and error. 
They are typically used in robotics – where a robot can learn to avoid collisions by receiving negative feedback after bumping into obstacles, and in video games – where trial and error reveals specific movements that can shoot up a player’s rewards. 
The agent can then use these rewards to understand the optimal state of game play and choose the next action.
 
Algorithms :

- Bagging with Random Forests.
- Boosting with AdaBoost.

### Overview
---

Ref : [Microsoft cheat sheet](https://docs.microsoft.com/en-us/azure/machine-learning/studio/algorithm-cheat-sheet)

![Chart](/data/img/ms-cheat.png)


### References :
---

- [Coursera - Andrew NG](https://www.coursera.org/learn/machine-learning)
- [Lecture_notes- Supervised Learning](http://cs229.stanford.edu/notes/cs229-notes1.pdf)
- [ML Cheat sheet](https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html#)
- [Digit Recognize](http://neuralnetworksanddeeplearning.com/chap1.html)
- [Own Notes](https://github.com/Bhanuchander210/Learn_MachineLearning)



### Others
---

- Sequence Prediction.
- Sequence Classification.
- Sequence Generation.
- Sequence to Sequence Prediction.

[Source](https://machinelearningmastery.com/sequence-prediction)