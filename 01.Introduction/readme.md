#### Machine Learning - A Quick Note

> "the field of study that gives computers the ability to learn without being explicitly programmed" by Arthur Samuel.

> "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E " by Tom Mitchell.

- [Supervised](#supervised-learning)
- [Unsupervised](#unsupervised-learning)

##### Supervised Learning

> With Data Sets or Observations

###### Types

- Classification
    In a classification problem, we are instead trying to predict 
    results in a discrete output. In other words, 
    we are trying to map input variables into discrete categories.
    
    **ex :** Given a patient with a tumor, we have to predict whether the tumor is malignant or benign. 
    
- Regression
    In a regression problem, we are trying to predict results within 
    a continuous output, meaning that 
    we are trying to map input variables to some continuous function.
    
    **ex :**  Given a picture of a person, we have to predict their age on the basis of the given picture.

##### Unsupervised Learning

> Without Data Sets or Observations

We don't have any idea what our results should look like and 
there is no feedback based on the prediction results.

###### Types

- **Clustering**
    Take a collection of 1,000,000 different genes, and find 
    a way to automatically group these genes into groups that 
    are somehow similar or related by different variables, 
    such as lifespan, location, roles, and so on.
    
- **Non-Clustering**
    The **Cocktail Party Algorithm**, allows you to find structure 
    in a chaotic environment. (i.e. identifying individual voices 
    and music from a mesh of sounds at a cocktail party).
