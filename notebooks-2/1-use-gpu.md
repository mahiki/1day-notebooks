
# GPUs

Check your CUDA driver and device. 


```python
!nvidia-smi
```

    /bin/sh: nvidia-smi: command not found


Number of available GPUs


```python
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

npx.num_gpus()
```




    0



Computation devices


```python
print(npx.cpu(), npx.gpu(), npx.gpu(1))

def try_gpu(i=0):
    return npx.gpu(i) if npx.num_gpus() >= i + 1 else npx.cpu()

def try_all_gpus():
    ctxes = [npx.gpu(i) for i in range(npx.num_gpus())]
    return ctxes if ctxes else [npx.cpu()]

try_gpu(), try_gpu(3), try_all_gpus()
```

    cpu(0) gpu(0) gpu(1)





    (cpu(0), cpu(0), [cpu(0)])



Create ndarrays on the 1st GPU


```python
x = np.ones((2, 3), ctx=try_gpu())
print(x.context)
x
```

    cpu(0)





    array([[1., 1., 1.],
           [1., 1., 1.]])



Create on the 2nd GPU


```python
y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
y
```




    array([[0.5488135 , 0.5928446 , 0.71518934],
           [0.84426576, 0.60276335, 0.8579456 ]])



Copying between devices


```python
z = x.copyto(try_gpu(1))
print(x)
print(z)
```

    [[1. 1. 1.]
     [1. 1. 1.]]
    [[1. 1. 1.]
     [1. 1. 1.]]


The inputs of an operator must be on the same device, then the computation will run on that device.


```python
y + z
```




    array([[1.59119  , 1.313164 , 1.7635204],
           [1.9731786, 1.3545473, 1.1167753]], ctx=gpu(1))



Initialize parameters on the first GPU.


```python
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

When the input is an ndarray on the GPU, Gluon will calculate the result on the same GPU.


```python
net(x)
```




    array([[0.04995865],
           [0.04995865]], ctx=gpu(0))



Let us confirm that the model parameters are stored on the same GPU.


```python
net[0].weight.data()
```




    array([[0.0068339 , 0.01299825, 0.0301265 ]], ctx=gpu(0))


