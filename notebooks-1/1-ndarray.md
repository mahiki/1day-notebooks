
# Data Manipulation with Ndarray

Importing `np` (numpy-like) module and `npx` (numpy extensions) module from MXNet. 


```python
from mxnet import np, npx
# Invoke the experimental numpy-compatible feature in MXNet 
npx.set_np()  
```

Create a vector and query its attributes


```python
x = np.arange(12)
x
```




    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.])




```python
x.shape
```




    (12,)




```python
x.size
```




    12



More ways to construct arrays


```python
np.zeros((3, 4))
```




    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])




```python
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```




    array([[2., 1., 4., 3.],
           [1., 2., 3., 4.],
           [4., 3., 2., 1.]])




```python
np.random.normal(0, 1, size=(3, 4))
```




    array([[ 2.2122064 ,  0.7740038 ,  1.0434405 ,  1.1839255 ],
           [ 1.8917114 , -1.2347414 , -1.771029  , -0.45138445],
           [ 0.57938355, -1.856082  , -1.9768796 , -0.20801921]])



Elemental-wise operators


```python
x = np.array([1, 2, 4, 8])
y = np.ones_like(x) * 2
print('x =', x)
print('x + y', x + y)
print('x - y', x - y)
print('x * y', x * y)
print('x ** y', x ** y)
print('x / y', x / y)
```

    x = [1. 2. 4. 8.]
    x + y [ 3.  4.  6. 10.]
    x - y [-1.  0.  2.  6.]
    x * y [ 2.  4.  8. 16.]
    x ** y [ 1.  4. 16. 64.]
    x / y [0.5 1.  2.  4. ]


Matrix multiplication.


```python
x = np.arange(12).reshape((3,4))
y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.dot(x, y.T)
```




    array([[ 18.,  20.,  10.],
           [ 58.,  60.,  50.],
           [ 98., 100.,  90.]])



Concatenate arrays along a particular axis. 


```python
np.concatenate([x, y], axis=0), np.concatenate([x, y], axis=1)
```




    (array([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.],
            [ 2.,  1.,  4.,  3.],
            [ 1.,  2.,  3.,  4.],
            [ 4.,  3.,  2.,  1.]]),
     array([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
            [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
            [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]]))



Broadcast Mechanism


```python
a = np.arange(3).reshape((3, 1))
b = np.arange(2).reshape((1, 2))
print('a:\n', a)
print('b:\n', b)
a + b
```

    a:
     [[0.]
     [1.]
     [2.]]
    b:
     [[0. 1.]]





    array([[0., 1.],
           [1., 2.],
           [2., 3.]])



Indexing and Slicing



```python
print('x[-1] =\n', x[-1])
print('x[1:3] =\n', x[1:3])
print('x[1:3, 2:4] =\n', x[1:3, 2:4])
print('x[1,2] =', x[1,2])
```

    x[-1] =
     [ 8.  9. 10. 11.]
    x[1:3] =
     [[ 4.  5.  6.  7.]
     [ 8.  9. 10. 11.]]
    x[1:3, 2:4] =
     [[ 6.  7.]
     [10. 11.]]
    x[1,2] = 6.0


`mxnet.numpy.ndarray` and `numpy.ndarray`


```python
a = x.asnumpy()
print(type(a))
b = np.array(a)
print(type(b))
```

    <class 'numpy.ndarray'>
    <class 'mxnet.numpy.ndarray'>

