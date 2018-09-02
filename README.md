# Implementations of logistic regression

### Notes about python 3
1. 1e-20 + 1 - 1 == 0; 1 - 1 + 1e-20 == 1e-20

### Notes about logistic regression
1. If you just use it without data normalization you have all zeros or all ones with zero gradient because of sigmoid function.
2. If you use too big gradient descent step (alpha) you have vacillating loss curve.
