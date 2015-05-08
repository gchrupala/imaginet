import theano
import theano.tensor as T
import passage.inits as inits
from theano.tensor.extra_ops import repeat
from passage.theano_utils import shared0s, floatX

class MatrixGroup(object):

    def __init__(self, n_features=256, sqrt_size=16, init=inits.uniform):
        self.n_features = n_features
        self.sqrt_size = sqrt_size
        self.size = self.sqrt_size ** 2
        self.init = init
        self.input = T.imatrix()
        self.embeddings = self.init((self.n_features, self.size))
        self.params = [self.embeddings]
        self.X0 = T.eye(self.sqrt_size, self.sqrt_size).reshape((1,self.size))

    def embedded(self):
        return self.embeddings[self.input]

    def step(self, x_t, h_tm1):
        h_t,_ = theano.scan(lambda x, z: T.dot(x, z), 
                            sequences=[  x_t.reshape((x_t.shape[0],   self.sqrt_size, self.sqrt_size)),
                                         h_tm1.reshape((h_tm1.shape[0], self.sqrt_size, self.sqrt_size))])
        return h_t.reshape((h_t.shape[0], self.size))

    def output(self, dropout_active=False):
        X = self.embedded()
        out, _ = theano.scan(self.step,
                             sequences=[X],
                             outputs_info=[repeat(self.X0, X.shape[1], axis=0)]
                         )
        return out[-1]

