
# Text Preprocessing


```python
import collections
import re
import random
from mxnet import np, npx
npx.set_np()
```

Read "Time Machine" by H. G. Wells as our training dataset


```python
def read_time_machine():
    with open('../data/timemachine.txt', 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line.strip().lower()) 
            for line in lines]

lines = read_time_machine()
'# sentences %d' % len(lines)
```




    '# sentences 3221'



Split each sentence into a list of tokens


```python
def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split(' ') for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unkown token type '+token)

tokens = tokenize(lines)
tokens[0:2]
```




    [['the', 'time', 'machine', 'by', 'h', 'g', 'wells', ''], ['']]



Build a vocabulary to map string tokens into numerical indices


```python
class Vocab(object):
    def __init__(self, tokens, min_freq=0):
        # Sort according to frequencies
        counter = collections.Counter([tk for line in tokens for tk in line])
        self.token_freqs = sorted(counter.items(), key=lambda x: x[0])
        self.token_freqs.sort(key=lambda x: x[1], reverse=True)
        self.unk, uniq_tokens = 0, ['<unk>']
        uniq_tokens +=  [token for token, freq in self.token_freqs 
                         if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1
    def __len__(self):
        return len(self.idx_to_token)
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
```

Print the map between a few tokens to indices


```python
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[0:10])
```

    [('<unk>', 0), ('the', 1), ('', 2), ('i', 3), ('and', 4), ('of', 5), ('a', 6), ('to', 7), ('was', 8), ('in', 9)]


Now we can convert each sentence into a list of numerical indices


```python
for i in range(8, 10):
    print('words:', tokens[i]) 
    print('indices:', vocab[tokens[i]])
```

    words: ['the', 'time', 'traveller', 'for', 'so', 'it', 'will', 'be', 'convenient', 'to', 'speak', 'of', 'him', '']
    indices: [1, 20, 72, 17, 38, 12, 120, 43, 706, 7, 660, 5, 112, 2]
    words: ['was', 'expounding', 'a', 'recondite', 'matter', 'to', 'us', 'his', 'grey', 'eyes', 'shone', 'and']
    indices: [8, 1654, 6, 3864, 634, 7, 131, 26, 344, 127, 484, 4]


Next load data into mini-batches


```python
def seq_data_iter_consecutive(corpus, batch_size, num_steps):
    # Offset for the iterator over the data for uniform starts
    offset = random.randint(0, num_steps)
    # Slice out data - ignore num_steps and just wrap around
    num_indices = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = np.array(corpus[offset:offset+num_indices])
    Ys = np.array(corpus[offset+1:offset+1+num_indices])
    Xs, Ys = Xs.reshape((batch_size, -1)), Ys.reshape((batch_size, -1))
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:,i:(i+num_steps)]
        Y = Ys[:,i:(i+num_steps)]
        yield X, Y
```

Test on a toy example


```python
my_seq = list(range(30))
for X, Y in seq_data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X =\n%s\nY =\n%s' %(X, Y))
```

    X =
    [[ 0.  1.  2.  3.  4.  5.]
     [14. 15. 16. 17. 18. 19.]]
    Y =
    [[ 1.  2.  3.  4.  5.  6.]
     [15. 16. 17. 18. 19. 20.]]
    X =
    [[ 6.  7.  8.  9. 10. 11.]
     [20. 21. 22. 23. 24. 25.]]
    Y =
    [[ 7.  8.  9. 10. 11. 12.]
     [21. 22. 23. 24. 25. 26.]]

