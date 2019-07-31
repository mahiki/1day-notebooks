
# Concise Implementation of Linear Regression


```python
import d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()

true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

Reading Data


```python
def load_array(data_arrays, batch_size, is_train=True):
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)
    
batch_size = 10
data_iter = load_array((features, labels), batch_size)
for X, y in data_iter:
    print('X =\n%sy =\n%s' %(X, y))
    break
```

    X =
    [[ 0.4015098   1.4096868 ]
     [ 0.65820086 -1.4260322 ]
     [ 0.00153129 -0.14330608]
     [-0.843129    0.6070013 ]
     [ 1.5080738  -0.27229312]
     [-0.01436996  0.50522786]
     [-0.2513225  -0.7733599 ]
     [-0.4892422   0.82852226]
     [ 0.19469471  0.26424283]
     [ 0.8269238   1.0562588 ]]y =
    [ 0.21267602 10.392891    4.7093544   0.45944032  8.140177    2.475876
      6.332503    0.4059622   3.69898     2.2603266 ]


Define the Model and initialize Model Parameters


```python
from mxnet.gluon import nn
from mxnet import init

net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))
```

Define the loss function and optimization algorithm


```python
from mxnet import gluon

loss = gluon.loss.L2Loss() 
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.03})
```

Training


```python
for epoch in range(1, 4):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean()))
    
w = net[0].weight.data()
print('Error in estimating w', true_w.reshape(w.shape) - w)
b = net[0].bias.data()
print('Error in estimating b', true_b - b)    
```

    epoch 1, loss: 0.040749
    epoch 2, loss: 0.000152
    epoch 3, loss: 0.000051
    Error in estimating w [[ 0.00024056 -0.00077081]]
    Error in estimating b [0.00041628]

