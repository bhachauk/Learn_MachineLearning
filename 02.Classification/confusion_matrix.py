# Example of a confusion matrix in Python
from sklearn.metrics import confusion_matrix

expected = ['cat','cat', 'dog', 'dog', 'dog','dog', 'rat', 'cat', 'rat', 'rat']
predicted = ['cat','dog', 'dog', 'dog', 'rat','dog', 'rat', 'cat', 'rat', 'dog']

results = confusion_matrix(expected, predicted)

print(results)