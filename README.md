# Learn_MachineLearning
Started from scratch stay tuned ....

## Basic Parameters
---

#### Mean, Median, Mode and Range

```python
import numpy as np

def mode(l):

	return max(set(l), key=l.count)

def range_of(l):

	return max(l)-min(l)

# Sample list

l = [1,3,5,7]

print 'List   :', l

print 'Mean   :',np.mean(l)

print 'Median :',np.median(l)

print 'Mode   :',mode (l)

print 'Range  :',range_of (l)
```

#### Detail - Description
---

```python
mean = (1 + 3 + 5 + 7)/4

print mean


variance = ((1-4)**2 + (3-4)**2 + (5-4)**2 +(7-5)** 2)/4

print variance


standard_deviation = variance ** (1/2) 

print standard_deviation
```

#### Why standard deviation (sigma) is 34.1 % 

Learn about the rule **68–95–99.7 Rule**

<p align="center">
Normal Distribution
<kbd>

![Rule](/data/img/rule-std.png)
</kbd> 
</p>




### Distributions

- Bernoulli Distribution
- Uniform Distribution
- Binomial Distribution
- Normal Distribution
- Poisson Distribution
- Exponential Distribution