
# Convolutions


```python
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

The cross-correlation operator.


```python
def corr2d(X, K):
    h, w = K.shape
    Y = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = np.array([[0, 1], [2, 3]])
corr2d(X, K)
```




    array([[19., 25.],
           [37., 43.]])



Convolutional layers


```python
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
```

Padding


```python
# A convenient function to test Gluon convoplution layers. 
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # Add batch and channel dimension.
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Exclude the first two dimensions
    return Y.reshape(Y.shape[2:])

conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = np.random.uniform(size=(8, 8))
comp_conv2d(conv2d, X).shape
```




    (8, 8)



Stride


```python
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```




    (4, 4)



A slightly more complicated example


```python
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```




    (2, 2)



Multiple input channels


```python
def corr2d_multi_in(X, K):
    return sum(corr2d(x, k) for x, k in zip(X, K))

X = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = np.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

corr2d_multi_in(X, K)
```




    array([[ 56.,  72.],
           [104., 120.]])



Multiple output channels


```python
def corr2d_multi_in_out(X, K):
    return np.stack([corr2d_multi_in(X, k) for k in K])

K = np.stack((K, K + 1, K + 2))
K.shape, corr2d_multi_in_out(X, K).shape
```




    ((3, 2, 2, 2), (3, 2, 2))


