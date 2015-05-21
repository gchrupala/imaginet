# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/gchrupala/repos/neuraltalk/imagernn/')
sys.path.append('/home/gchrupala/repos/Passage/')
import numpy
import data_provider as dp

data = dp.getDataProvider('flickr8k')

pairs = list(data.iterImageSentencePair(split='val'))#[0:1000]

from passage.preprocessing import Tokenizer

tokenizer = Tokenizer(min_df=5)

tokens = tokenizer.fit_transform([ pair['sentence']['raw'] for pair in pairs ])

#imgfeats =  [ pair['image']['feat'] for pair in pairs ]

import imaginet
from imaginet import *
imaginet = reload(imaginet)

from passage.layers import GatedRecurrent, Embedding,  OneHot


class Zeros(object):
    
    def __init__(self, size=128, weights=None):
        self.settings = locals()
        del self.settings['self']
        self.size  = size
        self.zeros = theano.shared(numpy.asarray(numpy.zeros((1,self.size)), dtype=theano.config.floatX))
        self.params = [self.zeros]
    
    def output(self, length=1, dropout_active=False):
        return repeat(self.zeros, length, axis=0)
        
class SharedEmbedding(object):

    def __init__(self, size=128, n_features=256, init='uniform', weights=None):
        self.settings = locals()
        del self.settings['self']
        self.init = getattr(inits, init)
        self.size = size
        self.n_features = n_features
        self.wv = self.init((self.n_features, self.size))
        self.params = [self.wv]

        if weights is not None:
            for param, weight in zip(self.params, weights):
                param.set_value(floatX(weight))

class EmbeddingOut(object):

    def __init__(self, embedding):
        self.settings = locals()
        del self.settings['self']
        self.embedding = embedding
        self.size = self.embedding.size
        self.input =  T.imatrix()
        self.params = []

    def output(self, dropout_active=False):
        return self.embedding.wv[self.input]
        

class GatedRecurrentWithH0(object):

    def __init__(self, size=256, activation='tanh', gate_activation='steeper_sigmoid', init='orthogonal', truncate_gradient=-1, seq_output=False, p_drop=0., weights=None):
        self.settings = locals()
        del self.settings['self']   
        self.activation_str = activation
        self.activation = getattr(activations, activation)
        self.gate_activation = getattr(activations, gate_activation)
        self.init = getattr(inits, init)
        self.size = size
        self.truncate_gradient = truncate_gradient
        self.seq_output = seq_output
        self.p_drop = p_drop
        self.weights = weights

    def connect(self, l_h0, l_in):
        self.l_in = l_in
        self.n_in = l_in.size
        self.l_h0 = l_h0

        self.w_z = self.init((self.n_in, self.size))
        self.w_r = self.init((self.n_in, self.size))

        self.u_z = self.init((self.size, self.size))
        self.u_r = self.init((self.size, self.size))

        self.b_z = shared0s((self.size))
        self.b_r = shared0s((self.size))

        if 'maxout' in self.activation_str:
            self.w_h = self.init((self.n_in, self.size*2)) 
            self.u_h = self.init((self.size, self.size*2))
            self.b_h = shared0s((self.size*2))
        else:
            self.w_h = self.init((self.n_in, self.size)) 
            self.u_h = self.init((self.size, self.size))
            self.b_h = shared0s((self.size))   

        self.params = [self.w_z, self.w_r, self.w_h, self.u_z, self.u_r, self.u_h, self.b_z, self.b_r, self.b_h]

        if self.weights is not None:
            for param, weight in zip(self.params, self.weights):
                param.set_value(floatX(weight))    


    def step(self, xz_t, xr_t, xh_t, h_tm1, u_z, u_r, u_h):
        z = self.gate_activation(xz_t + T.dot(h_tm1, u_z))
        r = self.gate_activation(xr_t + T.dot(h_tm1, u_r))
        h_tilda_t = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t = z * h_tm1 + (1 - z) * h_tilda_t
        return h_t

    def output(self, length=1, dropout_active=False):
        X = self.l_in.output(dropout_active=dropout_active).dimshuffle((1,0,2))
        H0 = self.l_h0.output(length=X.shape[1], dropout_active=dropout_active) 
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)
        x_z = T.dot(X, self.w_z) + self.b_z
        x_r = T.dot(X, self.w_r) + self.b_r
        x_h = T.dot(X, self.w_h) + self.b_h
        out, _ = theano.scan(self.step, 
            sequences=[x_z, x_r, x_h], 
            outputs_info=[H0], 
            non_sequences=[self.u_z, self.u_r, self.u_h],
            truncate_gradient=self.truncate_gradient
        )
        if self.seq_output:
            return out.dimshuffle((1,0,2))
        else:
            return out[-1]  



E = SharedEmbedding(size=256, n_features=tokenizer.n_features)
E_inp = EmbeddingOut(embedding=E)
E_out = EmbeddingOut(embedding=E)
H0 = Zeros(size=256)
H_enc = GatedRecurrentWithH0(seq_output=False, size=256)
H_dec = GatedRecurrentWithH0(seq_output=True,  size=256)
O = Dense(size=tokenizer.n_features, activation='softmax', reshape=True)
W = OneHot(n_features=tokenizer.n_features)



H_enc.connect(H0, E_inp)
H_dec.connect(H_enc, E_out)
O.connect(H_dec)

X = E_inp.input
Y_prev = E_out.input

import theano.tensor as T

Y_int = W.input

Y = W.output()

y_tr = O.output(dropout_active=True)
y_te = O.output(dropout_active=False)
cost = CategoricalCrossEntropySwapped(Y, y_tr)

from passage.updates import Adam

updater = Adam()

def flatten(l):
    return [item for sublist in l for item in sublist]

params = flatten([l.params for l in [E, E_inp, E_out, H0, H_enc, H_dec, O, W]]) 

updates = updater.get_updates(params, cost)

import theano

_train = theano.function([X, Y_prev, Y_int ], cost, updates=updates)
_predict = theano.function([X, Y_prev], y_te)


def pad(xss, padding):
    max_len = max((len(xs) for xs in xss))
    def pad_one(xs):
        return [ padding for _ in range(0,(max_len-len(xs))) ] + xs
    return [ pad_one(xs) for xs in xss ]


import numpy
inputs = numpy.array(pad([ sent + [tokenizer.encoder['END']] for sent in tokens ], tokenizer.encoder['PAD']), dtype='int32')
outputs = inputs #numpy.array(pad([ list(reversed(sent)) for sent in tokens ], tokenizer.encoder['PAD']), dtype='int32')

outputs_prev = numpy.array([ [tokenizer.encoder['PAD']] + list(output[:-1]) for output in outputs ], dtype='int32')

print inputs.shape
print outputs.shape
print outputs_prev.shape

mb_size = 64
N = len(inputs)
for it in range(1,5):
    j = 0
    while j < N:
        print it, j, _train(inputs[j:j+mb_size], outputs_prev[j:j+mb_size], outputs[j:j+mb_size])
        j = j + mb_size

for i in range(10):
    pred = _predict(inputs[i:i+1], outputs_prev[i:i+1])[0]
    print [ tokenizer.decoder[k] for k in inputs[i] ]
    print pred.shape, [ tokenizer.decoder[k] for k in numpy.argmax(pred, axis=1) ]
    print


