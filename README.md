# Learn_MachineLearning
Started from scratch stay tuned ....

### Basic Parameters

#### Mean, Median, Mode and Range

```python
import numpy as np

def mode(l):

	return max(set(l), key=l.count)

def range_of(l):

	return max(l)-min(l)

# Sample list

l = [1,3,5,7,7,9,10,34,22]

print 'List   :', l

print 'Mean   :',np.mean(l)

print 'Median :',np.median(l)

print 'Mode   :',mode (l)

print 'Range  :',range_of (l)
```
