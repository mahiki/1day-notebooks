
# A Hybrid of Imperative and Symbolic Programming


```python
import d2l
from mxnet import np, npx, sym
from mxnet.gluon import nn
npx.set_np()

def add(a, b):
    return a + b
def fancy_func(a, b, c):
    e = add(a, b)
    return add(c, e)
fancy_func(1, 2, 3)
```




    6



Symbolic programming


```python
def add_str():
    return '''def add(a, b):
    return a + b
'''
def fancy_func_str():
    return '''def fancy_func(a, b, c):
    e = add(a, b)
    return add(c, e)
'''
def evoke_str():
    return add_str() + fancy_func_str() + '''
print(fancy_func(1, 2, 3))
'''
prog = evoke_str()
y = compile(prog, '', 'exec')
exec(y)
```

    6


Construct with the ``HybridSequential`` class


```python
def get_net():
    net = nn.HybridSequential()
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = np.random.normal(size=(1, 512))
net = get_net()
net(x)
```




    array([[0.08827581, 0.00505182]])



Compile and optimize the workload


```python
net.hybridize()
net(x)
```




    array([[0.08827581, 0.00505182]])



Benchmark


```python
def benchmark(net, x):
    timer = d2l.Timer()
    for i in range(1000):
        _ = net(x)
    npx.waitall()
    return timer.stop()

net = get_net()
print('before hybridizing: %.4f sec' % (benchmark(net, x)))
net.hybridize()
print('after hybridizing: %.4f sec' % (benchmark(net, x)))
```

    before hybridizing: 0.5593 sec
    after hybridizing: 0.2651 sec


Export the program to other languages


```python
net.export('my_mlp')
!ls my_mlp*
!head -n20 my_mlp-symbol.json
```

    my_mlp-0000.params  my_mlp-symbol.json
    {
      "nodes": [
        {
          "op": "null", 
          "name": "data", 
          "inputs": []
        }, 
        {
          "op": "null", 
          "name": "dense3_weight", 
          "attrs": {
            "__dtype__": "0", 
            "__lr_mult__": "1.0", 
            "__shape__": "(256, -1)", 
            "__storage_type__": "0", 
            "__wd_mult__": "1.0"
          }, 
          "inputs": []
        }, 
        {

