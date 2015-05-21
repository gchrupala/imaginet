# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/gchrupala/repos/neuraltalk/imagernn/')
sys.path.append('/home/gchrupala/repos/Passage/')
import numpy
import theano.tensor as T
import theano

import data_provider as dp

data = dp.getDataProvider('flickr8k')

pairs = list(data.iterImageSentencePair(split='train'))

from passage.preprocessing import Tokenizer

tokenizer = Tokenizer(min_df=10)

tokens = tokenizer.fit_transform([ pair['sentence']['raw'] for pair in pairs ])

#imgfeats =  [ pair['image']['feat'] for pair in pairs ]

import imaginet
from imaginet import *
imaginet = reload(imaginet)

from passage.layers import GatedRecurrent, Embedding,  OneHot
from passage.updates import Adam

class TransposedDense(object):
    
    def __init__(self, layer, reshape=False):
        self.settings = locals()
        del self.settings['self']
        self.layer = layer
        self.reshape = reshape
        self.params = []
        self.n_in = self.layer.size
        self.size = self.layer.n_in
        
    def connect(self, l_in):
        self.l_in = l_in

    def output(self, dropout_active=False):
        X = self.l_in.output(dropout_active=dropout_active)
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)
        if self.reshape: #reshape for tensor3 softmax
            shape = X.shape
            X = X.reshape((shape[0]*shape[1], self.n_in))

        out =  self.layer.activation(T.dot(X, self.layer.w.T) + self.layer.b)

        if self.reshape: #reshape for tensor3 softmax
            out = out.reshape((shape[0], shape[1], self.size))

        return out

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


def flatten(l):
    return [item for sublist in l for item in sublist]

class EncoderDecoder(object):
    def __init__(self, embedding_size=128, size=128, vocab_size=128):
        self.embedding_size = embedding_size 
        self.size = size
        self.vocab_size = vocab_size
        self.E = SharedEmbedding(size=self.embedding_size, n_features=self.vocab_size)
        self.E_inp = EmbeddingOut(embedding=self.E)
        self.E_out = EmbeddingOut(embedding=self.E)
        self.H0 = Zeros(size=self.size)
        self.H_enc = GatedRecurrentWithH0(seq_output=False, size=self.size)
        self.H_dec = GatedRecurrentWithH0(seq_output=True,  size=self.size)
        self.O = Dense(size=self.vocab_size, activation='softmax', reshape=True)
        self.W = OneHot(n_features=self.vocab_size)

        self.H_enc.connect(self.H0, self.E_inp)
        self.H_dec.connect(self.H_enc, self.E_out)
        self.O.connect(self.H_dec)
        self.params = flatten([l.params for l in [self.E, self.E_inp, self.E_out, self.H0, self.H_enc, self.H_dec, self.O, self.W]]) 
        self.X = self.E_inp.input
        self.Y_prev = self.E_out.input
        self.Y_int = self.W.input

        self.Y = self.W.output()

        self.y_tr = self.O.output(dropout_active=True)
        self.y_te = self.O.output(dropout_active=False)
        self.cost = CategoricalCrossEntropySwapped(self.Y, self.y_tr)
        self.updater = Adam()
        self.updates = self.updater.get_updates(self.params, self.cost)
        self._train = theano.function([self.X, self.Y_prev, self.Y_int ], self.cost, updates=self.updates)
        self._predict = theano.function([self.X, self.Y_prev], self.y_te)

class EncoderDecoder2(object):
    def __init__(self, embedding_size=128, size=128, vocab_size=128):
        self.embedding_size = embedding_size 
        self.size = size
        self.vocab_size = vocab_size
        
        self.OH_inp = OneHot(n_features=self.vocab_size)
        self.OH_out = OneHot(n_features=self.vocab_size)
        # embedding
        self.E = Dense(size=self.embedding_size, activation='linear')
        self.E_inp = Wrapped(layer=self.E)
        self.E_out = Wrapped(layer=self.E)

        self.H0 = Zeros(size=self.size)
        self.H_enc = GatedRecurrentWithH0(seq_output=False, size=self.size)
        self.H_dec = GatedRecurrentWithH0(seq_output=True,  size=self.size)        
        # something to map hidden to embedding?
        ## FIXME TODO
        #self.O = Dense(size=self.vocab_size, activation='softmax', reshape=True)
        self.O = TransposedDense(layer=E)
        self.H_enc.connect(self.H0, self.E_inp)
        self.H_dec.connect(self.H_enc, self.E_out)
        self.O.connect(self.H_dec)
        
        self.params = flatten([l.params for l in [self.E, self.E_inp, self.E_out, self.H0, self.H_enc, self.H_dec, self.O, self.W]]) 
        self.X = self.E_inp.input
        self.Y_prev = self.E_out.input
        self.Y_int = self.W.input

        self.Y = self.W.output()

        self.y_tr = self.O.output(dropout_active=True)
        self.y_te = self.O.output(dropout_active=False)
        self.cost = CategoricalCrossEntropySwapped(self.Y, self.y_tr)
        self.updater = Adam()
        self.updates = self.updater.get_updates(self.params, self.cost)
        self._train = theano.function([self.X, self.Y_prev, self.Y_int ], self.cost, updates=self.updates)
        self._predict = theano.function([self.X, self.Y_prev], self.y_te)

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

model = EncoderDecoder(embedding_size=1024, size=1024, vocab_size=tokenizer.n_features)

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


