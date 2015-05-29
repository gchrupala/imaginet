# Mock interface for a composable layer
import numpy as np
import theano
import theano.tensor as T
# Utils

def shared0s(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)
def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)
def uniform(shape, scale=0.05):
    return sharedX(np.random.uniform(low=-scale, high=scale, size=shape))
def orthogonal(shape, scale=1.1):
    """ benanne lasagne ortho init (faster than qr approach)"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v # pick the one with the correct shape
    q = q.reshape(shape)
    return sharedX(scale * q[:shape[0], :shape[1]])
def tanh(x):
	return T.tanh(x)
def steeper_sigmoid(x):
	return 1./(1. + T.exp(-3.75 * x))
def softmax3d(inp):
    x = inp.reshape((inp.shape[0], inp.shape[1]*inp.shape[2]))
    e_x = T.exp(x - x.max(axis=1).dimshuffle(0, 'x'))
    result = e_x / e_x.sum(axis=1).dimshuffle(0, 'x')
    return result.reshape(inp.shape)

class Layer(object):

    def __init__(self):
        self.params = []

    def __call__(self, inp):
        raise NotImplementedError

    def compose(self, l2):
        l = Layer()
        l.__call__ = lambda inp: self(l2(inp))
        l.params = self.params + l2.params
        return l


class Embedding(Layer):
    
    def __init__(self, size_in, size_out):
        self.size_in = size_in
        self.size_out = size_out
        self.E = uniform((size_in, size_out))
        self.params = [self.E]

    def __call__(self, inp):
        return self.E[inp]

def theano_one_hot(idx, n):
    z = T.zeros((idx.shape[0], n))
    one_hot = T.set_subtensor(z[T.arange(idx.shape[0]), idx], 1)
    return one_hot        

class OneHot(Layer):

    def __init__(self, size_in):
        self.size_in = size_in
        self.params = []

    def __call__(self, inp):
        return theano_one_hot(inp.flatten(), self.size_in).reshape((inp.shape[0], inp.shape[1], self.size_in))
        

class Dense(Layer):
    
    def __init__(self, size_in, size_out):
        self.size_in = size_in
        self.size_out = size_out
        self.w = uniform((self.size_in, self.size_out))
        self.b = shared0s((self.size_out))
        self.params = [self.w, self.b]

    def __call__(self, inp):
        return T.dot(inp, self.w) + self.b

class GatedRecurrentWithH0(object):

    def __init__(self, size_in, size):
        self.size_in = size_in
        self.size = size
        self.activation = tanh
        self.gate_activation = steeper_sigmoid
        self.init = orthogonal
        self.size = size

        self.w_z = self.init((self.size_in, self.size))
        self.w_r = self.init((self.size_in, self.size))

        self.u_z = self.init((self.size, self.size))
        self.u_r = self.init((self.size, self.size))

        self.b_z = shared0s((self.size))
        self.b_r = shared0s((self.size))

        self.w_h = self.init((self.size_in, self.size)) 
        self.u_h = self.init((self.size, self.size))
        self.b_h = shared0s((self.size))   

        self.params = [self.w_z, self.w_r, self.w_h, self.u_z, self.u_r, self.u_h, self.b_z, self.b_r, self.b_h]

    def step(self, xz_t, xr_t, xh_t, h_tm1, u_z, u_r, u_h):
        z = self.gate_activation(xz_t + T.dot(h_tm1, u_z))
        r = self.gate_activation(xr_t + T.dot(h_tm1, u_r))
        h_tilda_t = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t = z * h_tm1 + (1 - z) * h_tilda_t
        return h_t

    def __call__(self, seq, h0, repeat_h0=0):
        X = seq.dimshuffle((1,0,2))
        H0 = T.repeat(h0, X.shape[1], axis=0) if repeat_h0 else h0
        x_z = T.dot(X, self.w_z) + self.b_z
        x_r = T.dot(X, self.w_r) + self.b_r
        x_h = T.dot(X, self.w_h) + self.b_h
        out, _ = theano.scan(self.step, 
            sequences=[x_z, x_r, x_h], 
                             outputs_info=[H0], 
            non_sequences=[self.u_z, self.u_r, self.u_h]
        )
        return out.dimshuffle((1,0,2))
        
class Zeros(Layer):
    
    def __init__(self, size):
        self.size  = size
        self.zeros = theano.shared(numpy.asarray(numpy.zeros((1,self.size)), dtype=theano.config.floatX))
        self.params = [self.zeros]
    
    def __call__(self):
        return self.zeros

def CategoricalCrossEntropySwapped(y_true, y_pred):
    return T.nnet.categorical_crossentropy(T.clip(y_pred, 1e-7, 1.0-1e-7), y_true).mean()
def MeanSquaredError(y_true, y_pred):
    return T.sqr(y_pred - y_true).mean()

import sys
sys.path.append('/home/gchrupala/repos/neuraltalk/imagernn/')
sys.path.append('/home/gchrupala/repos/Passage/')
import numpy
import theano.tensor as T
import theano
from passage.updates import Adam, SGD

import data_provider as dp

data = dp.getDataProvider('flickr8k')

pairs = list(data.iterImageSentencePair(split='train'))

from passage.preprocessing import Tokenizer

tokenizer = Tokenizer(min_df=10)

tokens = tokenizer.fit_transform([ pair['sentence']['raw'] for pair in pairs ])


    
class EncoderDecoder(object):
    def __init__(self, embedding_size, size, vocab_size):
        self.embedding_size = embedding_size 
        self.size = size
        self.vocab_size = vocab_size
        EMB = Embedding(self.vocab_size, self.embedding_size)
        ENC = GatedRecurrentWithH0(size_in=self.embedding_size, size=self.size)
        last = lambda x: x.dimshuffle((1,0,2))[-1]
        DEC = GatedRecurrentWithH0(size_in=self.embedding_size, size=self.size)
        H0  = Zeros(size=self.size)
        OUT = Dense(size_in=self.size, size_out=self.vocab_size) 
        OH  = OneHot(size_in=self.vocab_size)
        self.params = sum([ l.params for l in [EMB, ENC, DEC, H0, OUT, OH] ], [])
        self.MODEL = \
            lambda inp, out_prev: OUT(DEC(EMB(out_prev), last(ENC(EMB(inp), H0(), repeat_h0=1))))
        self.input       = T.imatrix()
        self.output_prev = T.imatrix()
        self.output      = T.imatrix('output')
        self.oh_output   = OH(self.output)
        self.y_tr = self.MODEL(self.input, self.output_prev)
        self.y_te = self.MODEL(self.input, self.output_prev)
        # FIXME need flattening?
        self.cost = CategoricalCrossEntropySwapped(self.oh_output, softmax3d(self.y_tr))
        #self.cost = MeanSquaredError(self.oh_output, self.y_tr)
        self.updater = Adam()
        self.updates = self.updater.get_updates(self.params, self.cost)
        self._train = theano.function([self.input, self.output_prev, self.output ], 
                                      self.cost, updates=self.updates)
        self._predict = theano.function([self.input, self.output_prev], self.y_te)

def pad(xss, padding):
    max_len = max((len(xs) for xs in xss))
    def pad_one(xs):
        return [ padding for _ in range(0,(max_len-len(xs))) ] + xs
    return [ pad_one(xs) for xs in xss ]

def main():
    import numpy
    inputs = numpy.array(pad([ sent + [tokenizer.encoder['END']] for sent in tokens ], tokenizer.encoder['PAD']), dtype='int32')
    outputs = inputs #numpy.array(pad([ list(reversed(sent)) for sent in tokens ], tokenizer.encoder['PAD']), dtype='int32')

    outputs_prev = numpy.array([ [tokenizer.encoder['PAD']] + list(output[:-1]) for output in outputs ], dtype='int32')

    print inputs.shape
    print outputs.shape
    print outputs_prev.shape

    model = EncoderDecoder(embedding_size=128, size=128, vocab_size=tokenizer.n_features)
    mb_size = 128
    N = len(inputs)
    for it in range(1,5):
        j = 0
        while j < N:
            print it, j, model._train(inputs[j:j+mb_size], outputs_prev[j:j+mb_size], outputs[j:j+mb_size])
            j = j + mb_size

        for i in range(100):
            pred = model._predict(inputs[i:i+1], outputs_prev[i:i+1])[0]
            print ' '.join(( tokenizer.decoder[k] for k in inputs[i] if k != tokenizer.encoder['PAD'] ))
            print pred.shape, ' '.join(( tokenizer.decoder[k] for k in numpy.argmax(pred, axis=1) if k != tokenizer.encoder['PAD']))
            print

main()

