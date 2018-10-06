# Introduction
---

We need to know all statistical parameters before starting to do Machine Learning.

Here I added some basic parameters which i met often. These are all used to understand the
data you have.

- [Basic parameters](#-basic-parameters)
- [Distributions](#-distributions)
- [DataVisualization](#data-visualization)



## Basic Parameters
---

#### Mean, Median, Mode and Range
---

Programming Language `python 2.7` used to explain the parameters.

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

#### Description in Python
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
---

Learn about the rule **68–95–99.7 Rule**

<p align="center">
<i>Normal Distribution</i>
</p>
<p align="center">
<kbd>
<img src="/data/img/rule-std.png"/></kbd> 
</p>

### Distributions
---

- [Bernoulli Distribution](#bernoulli-distribution)
- [Uniform Distribution](#uniform-distribution)
- [Binomial Distribution](#binomial-distribution)
- Normal Distribution
- Poisson Distribution
- Exponential Distribution

#### Bernoulli Distribution
---

x axis = Success Or Failure

y axis = Frequency (Number of Trials)

*Sample Code*
```python

import numpy as np
theta = 0.7
nobs = 10
Y = np.random.binomial(1, theta, nobs)

```

<p align="center">
<i>Bernoulli Distribution</i>
</p>
<p align="center">
<kbd>
<img src="/data/img/bernoulli.png" width="500" height="500"/></kbd> 
</p>

See Full code to plot [here](/01.Introduction/distributions/bernoulli.py)

#### Uniform Distribution
---

It can be called **Rectangular Distribution** or **Continous Distribution**.

***Example:***

*[Source](https://math.stackexchange.com/questions/1412305/real-world-examples-of-continuous-uniform-distribution-on-0-1)*

> You show up at a bus stop to wait for a bus that comes by once per hour. 
  You do not know what time the bus came by last. 
  The arrival time of the next bus is a continuous uniform distribution [0,1] measured in hours.


#### Binomial Distribution
---

x axis = Possible Results

y axis = Number of Success have

**Example 2 Trials with Dice :**

*Possible Results = 11*

*Total Number Of occurances (S) = 36* 

```
2{(1,1)}  => 1/36

3{(1,2),(2,1)} => 2/36

4{(2,2),(3,1),(1,3)} => 3/36

5{(1,4),(4,1),(2,3),(3,2)} => 4/36

6{(3,3),(1,5),(5,1),(2,4),(4,2)} => 5/36

7{(1,6),(6,1),(2,5),(5,2),(3,4),(4,3)} => 6/36

8{(2,6),(6,2),(3,5),(5,3),(4,4)} => 5/36

9{(3,6),(6,3),(5,4),(4,5)} => 4/36

10{(4,6),(6,4),(5,5)} => 3/36

11{(5,6),(6,5)} => 2/36

12{(6,6)} = > 1/36
```

<p align="center">
<i>Binomial Distribution</i>
</p>
<p align="center">
<kbd>
<img src="/data/img/binomial.png" width="500" height="500"/></kbd> 
</p>


Sources

- [https://www.kaggle.com/benhamner/python-data-visualizations](https://www.kaggle.com/benhamner/python-data-visualizations)

### Data Visualization

- [IRIS Data](/01.Introduction/datavisualization/iris-visualization.ipynb)
- [TITANIC Data](/01.Introduction/datavisualization/Nominal_Visualization_Titanic.ipynb)


**[BACK](/README.md)**