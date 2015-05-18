import sys
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor.extra_ops import repeat
import numpy as np
from time import time
import passage.costs
import passage.updates
import passage.iterators 
from passage.utils import case_insensitive_import, save
from passage.preprocessing import LenFilter, standardize_targets
from passage.utils import shuffle, iter_data
from passage.theano_utils import floatX, intX, shared0s
import passage.activations as activations
import passage.inits as inits
import cPickle
import gzip

def one_hot(X, n):
    X = np.asarray(X)
    Xoh = np.zeros(X.shape + (n,))
    d1 = np.repeat(np.arange(X.shape[0]), X.shape[1])
    d2 = np.tile(np.arange(X.shape[1]), X.shape[0])
    Xoh[d1, d2, X.flatten()] = 1.0
    return Xoh

def flatten(l):
    return [item for sublist in l for item in sublist]

class SortedPaddedXYZ(object):

    def __init__(self, size=64, shuffle=True, x_dtype=intX, y_dtype=floatX, z_dtype=floatX, size_y=128):
        self.size = size
        self.shuffle = shuffle
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.z_dtype = z_dtype
        self.size_y  = size_y

    def iterX(self, X):
        for x_chunk, chunk_idxs in iter_data(X, np.arange(len(X)), size=self.size*20):
            sort = np.argsort([len(x) for x in x_chunk])
            x_chunk = [x_chunk[idx] for idx in sort]
            chunk_idxs = [chunk_idxs[idx] for idx in sort]
            for xmb, idxmb in iter_data(x_chunk, chunk_idxs, size=self.size):
                xmb = padded(xmb, 0)
                yield self.x_dtype(xmb), idxmb   

    def iterXYZ(self, X, Y, Z):
        
        if self.shuffle:
            X, Y, Z = shuffle(X, Y, Z)

        for x_chunk, y_chunk, z_chunk in iter_data(X, Y, Z, size=self.size*20):
            sort = np.argsort([len(x) for x in x_chunk])
            x_chunk = [x_chunk[idx] for idx in sort]
            y_chunk = [y_chunk[idx] for idx in sort]
            z_chunk = [z_chunk[idx] for idx in sort]
            mb_chunks = [[x_chunk[idx:idx+self.size], 
                          y_chunk[idx:idx+self.size], 
                          z_chunk[idx:idx+self.size]]
                         for idx in range(len(x_chunk))[::self.size]]
            mb_chunks = shuffle(mb_chunks)
            for xmb, ymb, zmb in mb_chunks:
                xmb = padded(xmb, 0)
                y_zero = 0 
                ymb = one_hot(padded(ymb, y_zero), self.size_y)
                z_zero = [ 0 for _ in zmb[0] ]
                zmb = np.transpose(padded(zmb, z_zero)) #FIXME is this correct? Why?
                yield self.x_dtype(xmb), self.y_dtype(ymb), self.z_dtype(zmb) 

class ForkedRNN(object):

    def __init__(self, layers, cost_y, cost_z, alpha=0.5, updater='Adam', size_y=128, verbose=2,
                 interpolated=True, zero_shot=False):
        self.settings = locals()
        del self.settings['self']
        self.layers = layers

        self.cost_y = cost_y
        self.cost_z = cost_z

        if isinstance(updater, basestring):
            self.updater = case_insensitive_import(passage.updates, updater)()
        else:
            self.updater = updater
        self.iterator = SortedPaddedXYZ(size_y=size_y, shuffle=False)
        self.size_y = size_y
        self.verbose = verbose
        self.interpolated = interpolated
        self.zero_shot = zero_shot
        for i in range(1, len(self.layers)):
            self.layers[i].connect(self.layers[i-1])
        self.params = flatten([l.params for l in layers])
        self.alpha = alpha
        
        self.X = self.layers[0].input

        self.y_tr = self.layers[-1].output_left(dropout_active=True)
        self.y_te = self.layers[-1].output_left(dropout_active=False)
        self.Y = T.tensor3()

        self.z_tr = self.layers[-1].output_right(dropout_active=True)
        self.z_te = self.layers[-1].output_right(dropout_active=False)
        self.Z = T.matrix()
                                     
        cost_y = self.cost_y(self.Y, self.y_tr) 
        if self.zero_shot:  # In zero-shot setting, we disable z-loss for examples with zero z-targets
            cost_z = ifelse(T.gt(self.Z.norm(2), 0.0),  self.cost_z(self.Z, self.z_tr), 0.0) 
        else:
            cost_z = self.cost_z(self.Z, self.z_tr) 
        if self.interpolated:
            cost = self.alpha * cost_y + (1.0 - self.alpha) * cost_z
        else:
            cost = self.alpha * cost_y + cost_z
        cost_valid_y = self.cost_y(self.Y, self.y_te)
        cost_valid_z = self.cost_z(self.Z, self.z_te)
        cost_valid = self.alpha * cost_valid_y + (1.0 - self.alpha) * cost_valid_z
        
        self.updates = self.updater.get_updates(self.params, cost)
        #grads = theano.tensor.grad(cost, self.params)
        #norm = theano.tensor.sqrt(sum([theano.tensor.sum(g**2) for g in grads]))
        self._train = theano.function([self.X, self.Y, self.Z], cost, updates=self.updates)
        self._params = theano.function([], self.params[0])
        self._cost = theano.function([self.X, self.Y, self.Z], cost)
        self._cost_valid = theano.function([self.X, self.Y, self.Z], 
                                           [cost_valid_y, cost_valid_z, cost_valid])

        self._predict_y = theano.function([self.X], self.y_te)
        self._predict_z = theano.function([self.X], self.z_te)
        self._predict = theano.function([self.X], [self.y_te, self.z_te])

    def fit(self, trX, trY, trZ, batch_size=64, n_epochs=1, 
            snapshot_freq=1, path=None,
            valid=None):
        """Train model on given training examples and return the list of costs after each minibatch is processed.

        Args:
          trX (list) -- Inputs
          trY (list) -- Outputs
          batch_size (int, optional) -- number of examples in a minibatch (default 64)
          n_epochs (int, optional)  -- number of epochs to train for (default 1)
          len_filter (object, optional) -- object to filter training example by length (default LenFilter())
          snapshot_freq (int, optional) -- number of epochs between saving model snapshots (default 1)
          path (str, optional) -- prefix of path where model snapshots are saved.
            If None, no snapshots are saved (default None)
          valid (3-tuple, optional) -- validation data
        Returns:
          list -- costs of model after processing each minibatch
        """
        self.iterator.size = batch_size
        n = 0.
        stats = []
        t = time()
        costs = []
        for e in range(n_epochs):
            epoch_costs = []
            for xmb, ymb, zmb in self.iterator.iterXYZ(trX, trY, trZ):
                c = self._train(xmb, ymb, zmb)
                #c = out[0]
                #norm = out[1:]
                if np.isnan(c):
                    raise ValueError("Cost is NaN")
                epoch_costs.append(c)
                n += len(zmb)
                #print "norm: {0}".format(norm)
                if self.verbose >= 2:
                    n_per_sec = n / (time() - t)
                    n_left = len(trZ) - n % len(trZ)
                    time_left = n_left/n_per_sec
                    sys.stdout.write("\rEpoch %d Seen %d samples Avg cost %0.4f Time left %d seconds" % (e, n, np.mean(epoch_costs[-250:]), time_left))
                    sys.stdout.flush()
            costs.extend(epoch_costs)

            status = "Epoch %d Seen %d samples Avg cost %0.4f Time elapsed %d seconds" % (e, n, np.mean(epoch_costs[-250:]), time() - t)
            if self.verbose >= 2:
                sys.stdout.write("\r"+status) 
                sys.stdout.flush()
                sys.stdout.write("\n")
            elif self.verbose == 1:
                print status ; sys.stdout.flush()
            if path and e % snapshot_freq == 0:
                cPickle.dump(self, gzip.open("{0}.{1}".format(path, e),"w"))
            if valid:
                vaX, vaY, vaZ = valid
                costs_valid = [ self._cost_valid(x, y, z) for x,y,z 
                                in self.iterator.iterXYZ(vaX,vaY,vaZ) ]
                costs_valid_y, costs_valid_z, costs_valid_yz = zip(*costs_valid)
                print "{0:0.4f} {1:0.4f} {2:0.4f}".format(np.mean(costs_valid_y), 
                                                           np.mean(costs_valid_z), 
                                                           np.mean(costs_valid_yz))
                sys.stdout.flush()
                    
        return costs

    def predict_y(self, X):
        if isinstance(self.iterator, passage.iterators.Padded):
            return self.predict_iterator(X)
        elif isinstance(self.iterator, passage.iterators.SortedPadded):
            return self.predict_idxs(X)
        elif isinstance(self.iterator, SortedPaddedXYZ):
            return self.predict_y_unpad(X)
        else:
            raise NotImplementedError

    def predict_y_iterator(self, X):
        preds = []
        for xmb in self.iterator.iterX(X):
            pred = self._predict_y(xmb)
            preds.append(pred)
        return np.vstack(preds)

    def predict_y_idxs(self, X):
        preds = []
        idxs = []
        for xmb, idxmb in self.iterator.iterX(X):
            pred = self._predict_y(xmb)
            preds.append(pred)
            idxs.extend(idxmb)
        return np.vstack(preds)[np.argsort(idxs)]


    def predict_y_unpad(self, X):
        preds = []
        idxs =  []
        lens = map(len, X)
        for xmb, idxmb in self.iterator.iterX(X):
            pred = np.argmax(self._predict_y(xmb), axis=2).transpose()
            preds.append(pred)
            idxs.extend(idxmb)
        result = np.vstack(preds)[np.argsort(idxs)]
        return [ x[len(x)-leni:] for leni, x in zip(lens, result) ]

    def predict_z(self, X):
        preds = []
        idxs = []
        for xmb, idxmb in self.iterator.iterX(X):
            pred = self._predict_z(xmb)
            preds.append(pred)
            idxs.extend(idxmb)
        result = np.vstack(preds)[np.argsort(idxs)]
        return result

    def predict(self, X):
        return self.predict_z(X)

# FIXME this function is outside the class to keep pickle compatibility
# To be moved into the class
def predict_h(model, X):
    '''Extract last value of recurrent states.''' 
    _predict_h1 = theano.function([model.X], model.layers[1].left.bottom.output())
    _predict_h2 = theano.function([model.X], model.layers[1].right.bottom.output())
    preds_h1 = []
    preds_h2 = []
    idxs = []
    for xmb, idxmb in model.iterator.iterX(X):
        pred_h1 = _predict_h1(xmb)[-1] # keep only last one
        pred_h2 = _predict_h2(xmb)
        preds_h1.append(pred_h1)
        preds_h2.append(pred_h2)
        idxs.extend(idxmb)
    result_h1 = np.vstack(preds_h1)[np.argsort(idxs)]
    result_h2 = np.vstack(preds_h2)[np.argsort(idxs)]
    return (result_h1, result_h2)

# FIXME this function is outside the class to keep pickle compatibility
# To be moved into the class
def predict_h_simple(model, X):
    '''Extract last value of recurrent states.''' 
    _predict_h = theano.function([model.X], model.layers[1].output())
    preds_h = []
    idxs = []
    for xmb, idxmb in model.iterator.iterX(X):
        pred_h = _predict_h(xmb)
        preds_h.append(pred_h)
        idxs.extend(idxmb)
    result_h = np.vstack(preds_h)[np.argsort(idxs)]
    return result_h

def padded(seqs, zero):
    lens = map(len, seqs)
    max_len = max(lens)
    def pad(seq):
        return [ zero for j in range(0,max_len - len(seq))] + [ seq[i] for i in range(0, len(seq)) ]
    seqs_padded = np.asarray([ pad(seq) for seq in seqs ])
    axes = (1, 0) + tuple(range(2,len(seqs_padded.shape))) 
    return np.transpose(seqs_padded, axes=axes)

class RNN(object):

    def __init__(self, layers, cost, updater='Adam', verbose=2, Y=T.matrix(), iterator='SortedPadded'):
        self.settings = locals()
        del self.settings['self']
        self.layers = layers

        if isinstance(cost, basestring):
            self.cost = case_insensitive_import(costs, cost)
        else:
            self.cost = cost

        if isinstance(updater, basestring):
            self.updater = case_insensitive_import(updates, updater)()
        else:
            self.updater = updater

        if isinstance(iterator, basestring):
            self.iterator = getattr(iterators, iterator)()
        else:
            self.iterator = iterator

        self.verbose = verbose
        for i in range(1, len(self.layers)):
            self.layers[i].connect(self.layers[i-1])
        self.params = flatten([l.params for l in layers])

        self.X = self.layers[0].input
        self.y_tr = self.layers[-1].output(dropout_active=True)
        self.y_te = self.layers[-1].output(dropout_active=False)
        self.Y = Y
                                     
        cost = self.cost(self.Y, self.y_tr)
        cost_valid = self.cost(self.Y, self.y_te)
        self.updates = self.updater.get_updates(self.params, cost)

        self._train = theano.function([self.X, self.Y], cost, updates=self.updates)
        self._params = theano.function([], self.params[0])
        self._cost = theano.function([self.X, self.Y], cost)
        self._cost_valid = theano.function([self.X, self.Y], 
                                           [cost_valid])

        self._predict = theano.function([self.X], self.y_te)

    def fit(self, trX, trY, batch_size=64, n_epochs=1, len_filter=LenFilter(), snapshot_freq=1, 
            path=None, valid=None):
        """Train model on given training examples and return the list of costs after each minibatch is processed.

        Args:
          trX (list) -- Inputs
          trY (list) -- Outputs
          batch_size (int, optional) -- number of examples in a minibatch (default 64)
          n_epochs (int, optional)  -- number of epochs to train for (default 1)
          len_filter (object, optional) -- object to filter training example by length (default LenFilter())
          snapshot_freq (int, optional) -- number of epochs between saving model snapshots (default 1)
          path (str, optional) -- prefix of path where model snapshots are saved.
            If None, no snapshots are saved (default None)
          valid (2-tuple, optional) -- validation data  
        Returns:
          list -- costs of model after processing each minibatch
        """
        self.iterator.size = batch_size
        if len_filter is not None:
            trX, trY = len_filter.filter(trX, trY)
        trY = standardize_targets(trY, cost=self.cost)

        n = 0.
        stats = []
        t = time()
        costs = []
        for e in range(n_epochs):
            epoch_costs = []
            for xmb, ymb in self.iterator.iterXY(trX, trY):
                c = self._train(xmb, ymb)
                epoch_costs.append(c)
                n += len(ymb)
                if self.verbose >= 2:
                    n_per_sec = n / (time() - t)
                    n_left = len(trY) - n % len(trY)
                    time_left = n_left/n_per_sec
                    sys.stdout.write("\rEpoch %d Seen %d samples Avg cost %0.4f Time left %d seconds" % (e, n, np.mean(epoch_costs[-250:]), time_left))
                    sys.stdout.flush()
            costs.extend(epoch_costs)

            status = "Epoch %d Seen %d samples Avg cost %0.4f Time elapsed %d seconds" % (e, n, np.mean(epoch_costs[-250:]), time() - t)
            if self.verbose >= 2:
                sys.stdout.write("\r"+status) 
                sys.stdout.flush()
                sys.stdout.write("\n")
            elif self.verbose == 1:
                print status ; sys.stdout.flush()
            if path and e % snapshot_freq == 0:
                cPickle.dump(self, gzip.open("{0}.{1}".format(path, e),"w"))
            if valid:
                vaX, vaY = valid
                costs_valid = [ self._cost_valid(x, y) for x,y 
                                in self.iterator.iterXY(vaX,vaY) ]
                print "{0:0.4f}".format(np.mean(costs_valid))
                sys.stdout.flush()

        return costs

    def predict(self, X):
        if isinstance(self.iterator, passage.iterators.Padded):
            return self.predict_iterator(X)
        elif isinstance(self.iterator, passage.iterators.SortedPadded):
            return self.predict_idxs(X)
        else:
            raise NotImplementedError

    def predict_iterator(self, X):
        preds = []
        for xmb in self.iterator.iterX(X):
            pred = self._predict(xmb)
            preds.append(pred)
        return np.vstack(preds)

    def predict_idxs(self, X):
        preds = []
        idxs = []
        for xmb, idxmb in self.iterator.iterX(X):
            pred = self._predict(xmb)
            preds.append(pred)
            idxs.extend(idxmb)
        return np.vstack(preds)[np.argsort(idxs)]

def CategoricalCrossEntropySwapped(y_true, y_pred):
    return T.nnet.categorical_crossentropy(T.clip(y_pred, 1e-7, 1.0-1e-7), y_true).mean()

def CosineDistance(U, V):
        U_norm = U / U.norm(2,  axis=1).reshape((U.shape[0], 1))
        V_norm = V / V.norm(2, axis=1).reshape((V.shape[0], 1))
        W = (U_norm * V_norm).sum(axis=1)
        return (1 - W).mean()

class Combined(object):
    """Layer which combines two layers in parallel. Layer below should be a sequence.
    """
    
    def __init__(self, left, right, left_type='id', right_type='last', weights=None):
        """Create a Combined  layer. Layer below is a sequence: left_type and right_type
           specify which portion of the sequence the combined layers get as input.
        """
        self.settings = locals()
        del self.settings['self']
        self.left = left
        self.right = right
        self.left_type=left_type
        self.right_type=right_type
        self.weights = weights

    def connect(self, l_in):
        
        self.left.connect(WrappedLayer(l_in, self.left_type))
        self.right.connect(WrappedLayer(l_in, self.right_type))
        self.params = self.left.params + self.right.params
        if self.weights is not None:
            for param, weight in zip(self.params, self.weights):
                param.set_value(floatX(weight))            

    def output_left(self, dropout_active=False):
        return self.left.output(dropout_active=dropout_active)
        
    def output_right(self, dropout_active=False):
        return self.right.output(dropout_active=dropout_active)

class WrappedLayer(object):
    """Wrapped layer, with modified output."""
    def __init__(self, layer, output_type):
        self.settings = locals()
        del self.settings['self']
        self.layer = layer
        self.output_type = output_type
        self.size = layer.size
        
    def output(self, dropout_active=False):
        if self.output_type == 'id':
            return self.layer.output(dropout_active=dropout_active)
        elif self.output_type == 'last':
            return self.layer.output(dropout_active=dropout_active)[-1]
        else:
            raise ValueError("Unknown output type")

class Stacked(object):
    """Stack of connected layers."""
    def __init__(self, layers, weights=None):
        self.settings = locals()
        del self.settings['self']
        self.layers = layers
        self.bottom = self.layers[0]
        self.top = self.layers[-1]
        self.weights = weights

    def connect(self, l_in):
        self.bottom.connect(l_in)
        for i in range(1, len(self.layers)):
            self.layers[i].connect(self.layers[i-1])
        self.params = flatten([l.params for l in self.layers])        
        if self.weights is not None:
            for param, weight in zip(self.params, self.weights):
                param.set_value(floatX(weight))   

    def output(self, **kwargs):
        return self.top.output(**kwargs)

class Dense(object):
    def __init__(self, size=256, activation='rectify', init='orthogonal', p_drop=0., reshape=False, weights=None):
        self.settings = locals()
        del self.settings['self']
        self.activation_str = activation
        self.activation = getattr(activations, activation)
        self.init = getattr(inits, init)
        self.size = size
        self.p_drop = p_drop
        self.reshape = reshape
        self.weights = weights

    def connect(self, l_in):
        self.l_in = l_in
        self.n_in = l_in.size
        if 'maxout' in self.activation_str:
            self.w = self.init((self.n_in, self.size*2))
            self.b = shared0s((self.size*2))
        else:
            self.w = self.init((self.n_in, self.size))
            self.b = shared0s((self.size))
        self.params = [self.w, self.b]
        
        if self.weights is not None:
            for param, weight in zip(self.params, self.weights):
                param.set_value(floatX(weight))            

    def output(self, pre_act=False, dropout_active=False):
        X = self.l_in.output(dropout_active=dropout_active)
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)
        if self.reshape: #reshape for tensor3 softmax
            shape = X.shape
            X = X.reshape((shape[0]*shape[1], self.n_in))

        out =  self.activation(T.dot(X, self.w) + self.b)

        if self.reshape: #reshape for tensor3 softmax
            out = out.reshape((shape[0], shape[1], self.size))

        return out

