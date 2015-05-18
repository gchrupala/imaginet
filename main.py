#!/usr/bin/env python
from __future__ import division
import sys
sys.path.append('/home/gchrupala/repos/Passage')
sys.path.append('/home/gchrupala/repos/neuraltalk')
from passage.layers import Embedding, SimpleRecurrent, LstmRecurrent, GatedRecurrent #, Dense
from layers import *
from passage.costs import MeanSquaredError
from imaginet import *
from passage.preprocessing import Tokenizer, tokenize
import passage.utils
import passage.updates
from passage.iterators import SortedPadded
import imagernn.data_provider as dp
import cPickle
from scipy.spatial.distance import cosine, cdist
import numpy
import os.path
import argparse
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import json
import gzip

def main():
    parser = argparse.ArgumentParser(
        description='Learn to rank images according to similarity to \
                     caption meaning')
    parser.add_argument('--predict', dest='predict',
                        action='store_true', help='Run in prediction mode')
    parser.add_argument('--paraphrase', dest='paraphrase',
                        action='store_true', help='Run in paraphrasing mode')
    parser.add_argument('--paraphrase_state', dest='paraphrase_state', default='hidden_multi',
                        help='Which state to use for paraphrase retrieval (hidden_multi, hidden_vis, hidden_text, output_vis)')
    parser.add_argument('--extract_embeddings', dest='extract_embeddings',  
                        action='store_true',
                        help='Extract embeddings from trained model')
    parser.add_argument('--project_words', dest='project_words',
                        action='store_true', help='Project words from vocabulary to visual space')
    parser.add_argument('--model', dest='model', default='model.dat.gz',
                        help='Path to write model to')
    parser.add_argument('--model_type', dest='model_type', default='simple',
                        help='Type of model: (linear, simple, shared_embeddings, shared_all)')
    parser.add_argument('--character', dest='character', action='store_true', 
                        help='Character-level model')
    parser.add_argument('--zero_shot', dest='zero_shot', action='store_true',
                        help='Disable visual signal for sentences containing words in zero_shot.pkl.gz')
    parser.add_argument('--tokenizer', dest='tokenizer', default='tok.pkl.gz',
                        help='Path to write tokenizer to')
    parser.add_argument('--init_model', dest='init_model', default=None,
                        help='Initialize model weights with model from given path')
    parser.add_argument('--init_tokenizer', dest='init_tokenizer', default=None,
                        help='Use tokenizer from given path')
    parser.add_argument('--iter_predict', type=int,
                        help='Model after that many iterations will be used to predict')
    parser.add_argument('--scramble', action='store_true',
                        help='Scramble words in a test sentence')
    parser.add_argument('--distance', default='cosine',
                        help='Distance metric to rank images')
    parser.add_argument('--dataset', dest='dataset', default='flickr8k',
                        help='Dataset: flick8k, flickr30k, coco')
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=256,
                        help='size of the hidden layer')
    parser.add_argument('--embedding_size', dest='embedding_size', type=int, default=None,
                        help='size of (word) embeddings')
    parser.add_argument('--hidden_type', default='gru',
                        help='recurrent layer type: gru, lstm')
    parser.add_argument('--activation', default='tanh',
                        help='activation of the hidden layer units')
    parser.add_argument('--out_activation', default='linear',
                        help='Activation of output units')
    parser.add_argument('--cost', default='MeanSquaredError',
                        help='Image prediction cost function')
    parser.add_argument('--scaler', dest='scaler', default='none',
                        help='Method to scale targets (none, standard)')
    parser.add_argument('--rate', dest='rate', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--clipnorm', dest='clipnorm', type=float, default=0.0,
                        help='Gradients with norm larger than clipnorm will be scaled')
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.0,
                        help='Interpolation parameter for LM cost vs image cost')
    parser.add_argument('--ridge_alpha', dest='ridge_alpha', type=float, default=1.0,
                        help='Regularization for linear regression model')
    parser.add_argument('--non_interpolated', dest='non_interpolated', action='store_true',
                        help='Use non-interpolated cost')
    parser.add_argument('--iterations', dest='iterations', type=int, default=10,
                        help='Number of training iterations')
    parser.add_argument('--word_freq_threshold', dest='word_freq_threshold', type=int, default=10,
                        help='Map words below this threshold to UNK')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffle training data')
    parser.add_argument('--random_seed', dest='random_seed', default=None, type=int, 
                        help='Random seed')
    parser.add_argument('--snapshot_freq', dest='snapshot_freq', type=int, default=5,
                        help='How many iterations to save model')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64,
                        help='Batch size')
    args = parser.parse_args()
    if args.random_seed is not None:
        numpy.random.seed(args.random_seed)
    if args.project_words:
        project_words(args)
    elif args.predict and args.model_type == 'linear':
        test_linear(args)
    elif args.predict and args.model_type != 'linear':
        test(args)
    elif args.extract_embeddings:
        extract_embeddings(args)
    elif args.model_type == 'linear':
        train_linear(args)
    else:
        train(args)

class NoScaler():
    def __init__(self):
        pass
    def fit_transform(self, x):
        return x
    def transform(self, x):
        return x

def train_linear(args):
    p = dp.getDataProvider(args.dataset)
    data = list(p.iterImageSentencePair(split='train'))
    texts = [ pair['sentence']['raw'] for pair in data ]
    images = [ pair['image']['feat'] for pair in data ]
    analyzer = 'char' if args.character else 'word'
    vectorizer = CountVectorizer(min_df=args.word_freq_threshold, analyzer=analyzer, lowercase=True,
                                 ngram_range=(1,1))
    X = vectorizer.fit_transform(texts)
    scaler = StandardScaler() if args.scaler == 'standard' else NoScaler()
    sys.stderr.write("BOW computed\n")
    Y = scaler.fit_transform(numpy.array(images))
    
    
    model = Ridge(solver='lsqr', alpha=args.ridge_alpha)
    sys.stderr.write("Starting training\n")
    model.fit(X,Y)
    sys.stderr.write("Saving model\n")
    cPickle.dump(model, gzip.open('model.dat.gz','w'))
    cPickle.dump(vectorizer, gzip.open('vec.pkl.gz','w'))
    cPickle.dump(scaler, gzip.open('vec.pkl.gz', 'w'))

def test_linear(args):
    if args.random_seed is not None:
        numpy.random.seed(args.random_seed)
    D = Cdist()
    model = cPickle.load(gzip.open('model.dat.gz'))
    vectorizer = cPickle.load(gzip.open('vec.pkl.gz'))
    scaler = cPickle.load(gzip.open('scaler.pkl.gz'))
    real_stdout = sys.stdout
    with open('/dev/null', 'w') as f:
        sys.stdout = f
        d = dp.getDataProvider(args.dataset)
    sys.stdout = real_stdout
    pairs = list(d.iterImageSentencePair(split='val'))
    texts = [ pair['sentence']['raw'] for pair in pairs ]
    images    = list(d.iterImages(split='val')) # With pairs we'd get duplicate images!
    X = vectorizer.transform(texts)
    Y_pred = numpy.asarray(model.predict(X), dtype='float32') # candidates are identical to Y_pred
    if args.paraphrase:
        #distances = D.cosine_distance(Y_pred, Y_pred)
        distances = cdist(Y_pred, Y_pred, metric='cosine')
        N = 0
        score = 0.0
        for j,row in enumerate(distances):
            imgid = pairs[j]['sentence']['imgid']
            sentid = pairs[j]['sentence']['sentid']
            best = numpy.argsort(row)            
            top4 = sum([ imgid == pairs[b]['sentence']['imgid'] for b 
                         in best[0:5] if sentid != pairs[b]['sentence']['sentid'] ][0:4]) # exclude self
            score = score + top4/4.0
            N = N+1
        print args.iter_predict, N, score/N
 
    else:
        Y = numpy.array([ image['feat'] for image in images], dtype='float32')
        distances = D.cosine_distance(Y_pred, Y)
        errors = 0
        N = 0
        for j,row in enumerate(distances):
            imgid = pairs[j]['sentence']['imgid']
            best = numpy.argsort(row)
            top5 = [ images[b]['imgid'] for b in best[:5] ]
            N = N+1
            if imgid not in top5:
                errors = errors + 1
        print errors, N, errors/N

def train(args):
    zero_words = cPickle.load(gzip.open("zero_shot.pkl.gz")) if args.zero_shot else set()
    def maybe_zero(s, i):
        overlap = set(tokenize(s)).intersection(zero_words)    
        if args.zero_shot and len(overlap) > 0:
            return numpy.zeros(i.shape)
        else:
            return i
    dataset = args.dataset
    tok_path = args.tokenizer
    model_path = args.model 
    d = dp.getDataProvider(dataset)
    pairs = list(d.iterImageSentencePair(split='train'))
    if args.shuffle:
        numpy.random.shuffle(pairs)
    output_size = len(pairs[0]['image']['feat'])
    embedding_size = args.embedding_size if args.embedding_size is not None else args.hidden_size
    tokenizer = cPickle.load(gzip.open(args.init_tokenizer)) \
                    if args.init_tokenizer else Tokenizer(min_df=args.word_freq_threshold, character=args.character)
    sentences, images = zip(*[ (pair['sentence']['raw'], maybe_zero(pair['sentence']['raw'],pair['image']['feat']))
                               for pair in pairs ])
    scaler = StandardScaler() if args.scaler == 'standard' else NoScaler()
    images = scaler.fit_transform(images)
    tokens = [ [tokenizer.encoder['PAD']] + sent + [tokenizer.encoder['END'] ]
               for sent in tokenizer.fit_transform(sentences) ]
    tokens_inp = [ token[:-1] for token in tokens ]

    tokens_out = [ token[1:]  for token in tokens ]

    cPickle.dump(tokenizer, gzip.open(tok_path, 'w'))
    cPickle.dump(scaler, gzip.open('scaler.pkl.gz','w'))
    # Validation data
    valid_pairs = list(d.iterImageSentencePair(split='val'))
    valid_sents, valid_images  = zip(*[ (pair['sentence']['raw'], pair['image']['feat'])
                                        for pair in valid_pairs ])
    valid_images = scaler.transform(valid_images)
    valid_tokens = [ [ tokenizer.encoder['PAD'] ] + sent + [tokenizer.encoder['END'] ] 
                       for sent in tokenizer.transform(valid_sents) ]
    valid_tokens_inp = [ token[:-1] for token in valid_tokens ]
    valid_tokens_out = [ token[1:] for token in valid_tokens ]
    valid = (valid_tokens_inp, valid_tokens_out, valid_images)

    updater = passage.updates.Adam(lr=args.rate, clipnorm=args.clipnorm)
    if args.cost == 'MeanSquaredError':
        z_cost = MeanSquaredError
    elif args.cost == 'CosineDistance':
        z_cost = CosineDistance

    if args.hidden_type == 'gru':
        Recurrent = GatedRecurrent
    elif args.hidden_type == 'lstm':
        Recurrent = LstmRecurrent
    else:
        Recurrent = GatedRecurrent
    # if args.init_model is not None:
    #     model_init =  cPickle.load(open(args.init_model))
        
    #     def values(ps):
    #         return [ p.get_value() for p in ps ]
    #     # FIXME enable this for shared only embeddings 
    #     layers = [  Embedding(size=args.hidden_size, n_features=tokenizer.n_features, 
    #                           weights=values(model_init.layers[0].params)), 
    #                 Recurrent(seq_output=True, size=args.hidden_size, activation=args.activation,
    #                                weights=values(model_init.layers[1].params)),
    #                 Combined(left=Dense(size=tokenizer.n_features, activation='softmax', reshape=True,
    #                                     weights=values(model_init.layers[2].left.params)), 
    #                          right=Dense(size=output_size, activation=args.out_activation, 
    #                                      weights=values(model_init.layers[2].right.params))
    #                                  ) ]
        
    # else:
    # FIXME implement proper pretraining FIXME
    interpolated = True if not args.non_interpolated else False
    if args.model_type in ['add', 'mult', 'matrix']:
        if args.model_type == 'add':
            layer0 = Direct(size=embedding_size, n_features=tokenizer.n_features, op=Add)
        elif args.model_type == 'mult':
            layer0 = Direct(size=embedding_size, n_features=tokenizer.n_features, op=Mult)
        elif args.model_type == 'matrix':
            sqrt_size = embedding_size ** 0.5
            if not sqrt_size.is_integer():
                raise ValueError("Sqrt of embedding_size not integral for matrix model")
            layer0 = Direct(size=embedding_size, n_features=tokenizer.n_features, op=MatrixMult)
        layers = [ layer0, Dense(size=output_size, activation=args.out_activation, reshape=False) ]
        valid = (valid_tokens_inp, valid_images)
        model = RNN(layers=layers, updater=updater, cost=z_cost, 
                    iterator=SortedPadded(shuffle=False), verbose=1)
        model.fit(tokens_inp, images, n_epochs=args.iterations, batch_size=args.batch_size, len_filter=None,
                  snapshot_freq=args.snapshot_freq, path=model_path, valid=valid)
    elif args.model_type   == 'simple':
        layers = [ Embedding(size=embedding_size, n_features=tokenizer.n_features),
                   Recurrent(seq_output=False, size=args.hidden_size, activation=args.activation),
                   Dense(size=output_size, activation=args.out_activation, reshape=False)
                 ]
        valid = (valid_tokens_inp, valid_images)
        model = RNN(layers=layers, updater=updater, cost=z_cost, 
                    iterator=SortedPadded(shuffle=False), verbose=1)
        model.fit(tokens_inp, images, n_epochs=args.iterations, batch_size=args.batch_size, len_filter=None,
                  snapshot_freq=args.snapshot_freq, path=model_path, valid=valid)
        # FIXME need validation
    elif args.model_type   == 'deep-simple':
        layers = [ Embedding(size=embedding_size, n_features=tokenizer.n_features),
                   Recurrent(seq_output=True,  size=args.hidden_size, activation=args.activation),
                   Recurrent(seq_output=False, size=args.hidden_size, activation=args.activation),
                   Dense(size=output_size, activation=args.out_activation, reshape=False)
                 ]
        valid = (valid_tokens_inp, valid_images)
        model = RNN(layers=layers, updater=updater, cost=z_cost, 
                    iterator=SortedPadded(shuffle=False), verbose=1)
        model.fit(tokens_inp, images, n_epochs=args.iterations, batch_size=args.batch_size, len_filter=None,
                  snapshot_freq=args.snapshot_freq, path=model_path, valid=valid)
        # FIXME need validation
        
    elif args.model_type == 'shared_all':
        if args.zero_shot:
            raise NotImplementedError # FIXME zero_shot not implemented
        layers = [  Embedding(size=embedding_size, n_features=tokenizer.n_features), 
                    Recurrent(seq_output=True, size=args.hidden_size, activation=args.activation),
                    Combined(left=Dense(size=tokenizer.n_features, activation='softmax', reshape=True), 
                             right=Dense(size=output_size, activation=args.out_activation, reshape=False)) ] 

        model = ForkedRNN(layers=layers, updater=updater, cost_y=CategoricalCrossEntropySwapped, 
                          cost_z=z_cost, alpha=args.alpha, size_y=tokenizer.n_features, 
                          verbose=1, interpolated=interpolated) 

        model.fit(tokens_inp, tokens_out, images, n_epochs=args.iterations, batch_size=args.batch_size,
                  snapshot_freq=args.snapshot_freq, path=model_path, valid=valid)
    elif args.model_type == 'shared_embeddings':
        layers = [ Embedding(size=embedding_size, n_features=tokenizer.n_features),
                   Combined(left=Stacked([Recurrent(seq_output=True, size=args.hidden_size, activation=args.activation), 
                                          Dense(size=tokenizer.n_features, activation='softmax', reshape=True)]), 
                            left_type='id',
                            right=Stacked([Recurrent(seq_output=False, size=args.hidden_size, activation=args.activation), 
                                           Dense(size=output_size, activation=args.out_activation, reshape=False)]),
                            right_type='id')
                        ]

        model = ForkedRNN(layers=layers, updater=updater, cost_y=CategoricalCrossEntropySwapped, 
                          cost_z=z_cost, alpha=args.alpha, size_y=tokenizer.n_features, 
                          verbose=1, interpolated=interpolated, zero_shot=args.zero_shot)

        model.fit(tokens_inp, tokens_out, images, n_epochs=args.iterations, batch_size=args.batch_size,
                  snapshot_freq=args.snapshot_freq, path=model_path, valid=valid)

    cPickle.dump(model, gzip.open(model_path,"w"))


def test(args):
    if args.random_seed is not None:
        numpy.random.seed(args.random_seed)
    def scramble(words):
        ixs = range(len(words))
        random.shuffle(ixs)
        return [ words[ix] for ix in ixs ]
    testInfo = {'argv':       sys.argv,
                'dataset':    args.dataset,
                'scramble':   args.scramble,
                'model_type': args.model_type,
                'alpha':      args.alpha,
                'iter_predict': args.iter_predict,
                'task':       'paraphrase' if args.paraphrase else 'image',
                'items':      []}
    D = Cdist()
    dataset = args.dataset
    suffix = '' if args.iter_predict is None else ".{0}".format(args.iter_predict)
    model = cPickle.load(gzip.open('model.dat.gz' + suffix))
    tokenizer = cPickle.load(gzip.open('tok.pkl.gz'))
    scaler = cPickle.load(gzip.open('scaler.pkl.gz'))
    real_stdout = sys.stdout
    with open('/dev/null', 'w') as f:
        sys.stdout = f
        d = dp.getDataProvider(args.dataset)
    sys.stdout = real_stdout
    pairs = list(d.iterImageSentencePair(split='val'))
    inputs = [ scramble(s) if args.scramble else s for s in tokenizer.transform([ pair['sentence']['raw'] for pair in pairs]) ]
    if args.paraphrase:
        candidates = tokenizer.transform([ pair['sentence']['raw'] for pair in pairs]) # No scrambling of candidates
        if   args.paraphrase_state == 'output_vis':
            preds           = model.predict(inputs)
            candidates_pred = model.predict(candidates)
        elif args.paraphrase_state == 'hidden_text':
            preds, _           = predict_h(model, inputs) 
            candidates_pred, _ = predict_h(model, candidates)
        elif args.paraphrase_state == 'hidden_vis' and hasattr(model.layers[1], 'left'):
            _, preds           = predict_h(model, inputs)
            _, candidates_pred = predict_h(model, candidates)
        elif args.paraphrase_state == 'hidden_vis' and not hasattr(model.layers[1], 'left'):
            preds           = predict_h_simple(model, inputs)
            candidates_pred = predict_h_simple(model, candidates)
        elif args.paraphrase_state == 'hidden_multi':
            preds           = numpy.hstack(predict_h(model, inputs))
            candidates_pred = numpy.hstack(predict_h(model, candidates))
        else:
            raise ValueError("Unknown state")

        distances = D.cosine_distance(preds, candidates_pred)
        #distances = cdist(preds, candidates_pred, metric='cosine')
        N = 0
        score = 0.0
        imgids = numpy.array([ pair['sentence']['imgid'] for pair in pairs ])
        sentids = numpy.array([ pair['sentence']['sentid'] for pair in pairs])
        for j,row in enumerate(distances):
            imgid = pairs[j]['sentence']['imgid']
            sentid = pairs[j]['sentence']['sentid']
            best = numpy.argsort(row)
            rank = numpy.where((imgids[best] == imgid) * (sentids[best] != sentid))[0][0] + 1
            top4 = [ pairs[b]['sentence']['imgid'] for b 
                         in best[0:5] if sentid != pairs[b]['sentence']['sentid'] ][0:4] # exclude self
            top4sent = [ pairs[b]['sentence']['sentid'] for b in best[0:5] if sentid != pairs[b]['sentence']['sentid'] ][0:4]
            score = score + sum([i == imgid for i in top4 ])/4.0
            N = N+1
            itemInfo = {'sentid':sentid, 'imgid': imgid, 'score': sum([i == imgid for i in top4 ])/4.0, 
                        'rank': rank, 'topn': top4 , 'topnsentid': top4sent,
                        'input': tokenizer.inverse_transform([inputs[j]])[0]}
            testInfo['items'].append(itemInfo)
        print args.iter_predict, N, score/N
    else:
        preds     = model.predict(inputs)
        images    = list(d.iterImages(split='val')) 
        distances = D.cosine_distance(preds, scaler.transform([image['feat'] for image in images ]))
        errors = 0
        N = 0
        imgids = numpy.array([ img['imgid'] for img in images ])
        for j,row in enumerate(distances):
            imgid = pairs[j]['sentence']['imgid']
            sentid = pairs[j]['sentence']['sentid']
            best = numpy.argsort(row)
            rank = numpy.where(imgids[best] == imgid)[0][0] + 1
            top5 = [ images[b]['imgid'] for b in best[:5] ]
            N = N+1
            if imgid not in top5:
                errors = errors + 1
            itemInfo = {'sentid':sentid, 'imgid': imgid, 'score': float(imgid in top5), 'rank': rank, 'topn': top5, 
                        'input':tokenizer.inverse_transform([inputs[j]])[0] }
            testInfo['items'].append(itemInfo)
        print args.iter_predict, errors, N, errors/N
    testInfoPath = 'testInfo-task={0}-scramble={1}-iter_predict={2}.json.gz'.format(testInfo['task'], testInfo['scramble'], testInfo['iter_predict'])
    json.dump(testInfo, gzip.open(testInfoPath,'w'))

def project_words(args):
    suffix = '' if args.iter_predict is None else ".{0}".format(args.iter_predict)
    model = cPickle.load(gzip.open('model.dat.gz' + suffix))
    tokenizer = cPickle.load(gzip.open('tok.pkl.gz'))
    scaler = cPickle.load(gzip.open('scaler.pkl.gz'))
    exclude = ['PAD','END','UNK']
    words, indexes = zip(*[ (w,i) for (w,i) in tokenizer.encoder.iteritems() if w not in exclude ])
    inputs = [ [tokenizer.encoder['PAD'], i, tokenizer.encoder['END']] for i in indexes ]
    preds  = scaler.inverse_transform(model.predict(inputs))
    proj = dict((words[i], preds[i]) for i in range(0, len(words)))
    cPickle.dump(proj, gzip.open("proj.pkl.gz" + suffix, "w"))
    
def extract_embeddings(args):
    tokenizer = cPickle.load(gzip.open('tok.pkl.gz'))
    #scaler = cPickle.load(open('scaler.pkl'))
    suffix = '' if args.iter_predict is None else ".{0}".format(args.iter_predict)
    model = cPickle.load(gzip.open('model.dat.gz' + suffix))
    embeddings = model.layers[0].params[0].get_value()
    table = dict((word, embeddings[i]) for i,word in tokenizer.decoder.iteritems() 
                 if word not in ['END','PAD','UNK'] )
    cPickle.dump(table, gzip.open('embeddings.pkl.gz' + suffix, 'w'))

class Cdist():
    def __init__(self):
        self.U = T.matrix('U')
        self.V = T.matrix('V')
        self.U_norm = self.U / self.U.norm(2, axis=1).reshape((self.U.shape[0], 1))
        self.V_norm = self.V / self.V.norm(2, axis=1).reshape((self.V.shape[0], 1))
    
        self.W = T.dot(self.U_norm, self.V_norm.T)
        self.cosine = theano.function([self.U, self.V], self.W)

    def cosine_distance(self, A, B):
        return 1 - self.cosine(A, B)

main()
