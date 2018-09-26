#### What is Confusion Matrix

It is the summary metric of classification model. It defines about confusion level
of our classification model.

```text
Expected, 	Predicted
cat,		cat     pass
cat, 		dog     fail
dog,		dog     pass
dog,		dog     pass
dog,		rat     fail
dog, 		dog     pass
rat, 		rat     pass
cat, 		cat     pass
rat, 		rat     pass
rat, 		dog     fail
``` 

###### Accuracy
---

```text
accuracy = total correct predictions / total predictions made * 100
accuracy = 7 / 10 * 100
```

###### Prediction Analogy
---

**Correctly Predicted :**

```text
cat - cat 2
dog - dog 3
rat - rat 2
```

**Wrongly Predicted:**

```text
cat - dog   1
cat - rat   0

dog - cat   0
dog - rat   

```

**Anology:**

```text
    cat    dog    rat
    
cat  2      1      0

dog  0      3       1

rat  0      1       2

```


This is the confusion matrix.


