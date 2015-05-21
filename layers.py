import theano
import theano.tensor as T
import passage.inits as inits
from theano.tensor.extra_ops import repeat
from passage.theano_utils import shared0s, floatX


### Directly compositional models

### Associative operators ###

class Add(object):
    '''Elementwise addition.'''
    def __init__(self, size):
        self.size = size
        self.id = T.alloc(0.0, 1, self.size)

    def step(self, x_t, h_tm1):
        return h_tm1 + x_t

class Mult(object):
    '''Elementwise multiplication.'''
    def __init__(self, size):
        self.size = size
        self.id = T.alloc(1.0, 1, self.size)

    def step(self, x_t, h_tm1):
        return h_tm1 * x_t

class MatrixMult(object):
    '''Matrix multiplication.'''
    def __init__(self, size):
        self.size = size
        self.sqrt_size = int(self.size**0.5)
        self.id = T.eye(self.sqrt_size, self.sqrt_size).reshape((1,self.size))

    def step(self, x_t, h_tm1):
        h_t,_ = theano.scan(lambda x, z: T.dot(x, z), 
                            sequences=[  x_t.reshape((x_t.shape[0],   self.sqrt_size, self.sqrt_size)),
                                         h_tm1.reshape((h_tm1.shape[0], self.sqrt_size, self.sqrt_size))])
        return h_t.reshape((h_t.shape[0], self.size))
        
class Direct(object):

    def __init__(self, n_features=256, size=256, init=inits.uniform, op=MatrixMult):
        self.n_features = n_features
        self.size = size
        self.sqrt_size = int(self.size ** 0.5)
        self.init = init
        self.op = op(self.size)
        self.input = T.imatrix()
        self.embeddings = self.init((self.n_features, self.size))
        self.params = [self.embeddings]


    def embedded(self):
        return self.embeddings[self.input]

    def output(self, dropout_active=False):
        X = self.embedded()
        out, _ = theano.scan(self.op.step,
                             sequences=[X],
                             outputs_info=[repeat(self.op.id, X.shape[1], axis=0)]
                         )
        return out[-1]
