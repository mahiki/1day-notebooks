
# Automatic Differentiation


```python
from mxnet import autograd, np, npx  
npx.set_np()

x = np.arange(4)
x
```




    array([0., 1., 2., 3.])



Allocate space to store the gradient with respect to ``x``.


```python
x.attach_grad()
```

Record the computation within the `record` scope.


```python
with autograd.record():
    y = 2.0 * np.dot(x, x)
y
```




    array(28.)



The gradient of the function $y = 2\mathbf{x}^{\top}\mathbf{x}$ with respect to $\mathbf{x}$ should be $4\mathbf{x}$. 


```python
y.backward()
x.grad - 4 * x
```




    array([0., 0., 0., 0.])


