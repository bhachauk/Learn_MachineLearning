##### Normal Equation:

```equation
θ =(XT X)^ −1 XT y
```


- For larger sets , It will turn big matrix and needs more processing time. Choosing gradient descend method is good for larger data sets.

In details:

|Gradient Descent|Normal Equation|
|----------------|---------------|
|Need to choose alpha	| No need to choose alpha|
|Needs many iterations	|No need to iterate|
|O (kn^2kn 2)| O (n^3n 3), need to calculate inverse of XT X|
|Works well when n is large	| Slow if n is very large|
