
# Pooling


```python
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

Implement 2-d pooling


```python
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = np.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = np.max(X[i: i + p_h, j: j + p_w])
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
pool2d(X, (2, 2))
```




    array([[4., 5.],
           [7., 8.]])



Padding and Stride


```python
X = np.arange(16).reshape((1, 1, 4, 4))
print(X)

pool2d = nn.MaxPool2D(3)
pool2d(X)
```

    [[[[ 0.  1.  2.  3.]
       [ 4.  5.  6.  7.]
       [ 8.  9. 10. 11.]
       [12. 13. 14. 15.]]]]





    array([[[[10.]]]])



Specify the padding and stride


```python
print(X)
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

    [[[[ 0.  1.  2.  3.]
       [ 4.  5.  6.  7.]
       [ 8.  9. 10. 11.]
       [12. 13. 14. 15.]]]]





    array([[[[ 5.,  7.],
             [13., 15.]]]])



Multiple channels


```python
X = np.concatenate((X, X + 1), axis=1)
print(X)
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

    [[[[ 0.  1.  2.  3.]
       [ 4.  5.  6.  7.]
       [ 8.  9. 10. 11.]
       [12. 13. 14. 15.]]
    
      [[ 1.  2.  3.  4.]
       [ 5.  6.  7.  8.]
       [ 9. 10. 11. 12.]
       [13. 14. 15. 16.]]]]





    array([[[[ 5.,  7.],
             [13., 15.]],
    
            [[ 6.,  8.],
             [14., 16.]]]])


