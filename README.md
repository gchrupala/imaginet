# imaginet

Imaginet implements several models which read sentences describing
images and learn to build representations of these images grounded in
the visual features of the corresponding images.



Installation
============

Currently, installation is a manual. The main prerequisite is
theano. Additionally, you will also need
https://github.com/gchrupala/Passage. If you want to train new models
on the Flickr30k or MSCOCO datasets, you'll also need
https://github.com/gchrupala/neuraltalk. You can simply add these
libraries to the PYTHONPATH environment variable.

Usage 
=====

Predict
-------


You can use a pre-trained imaginet model to project sentences to the
space of visual features. For example, given the model stored in
data/multitask:

```python
from imaginet import *
workflow = load_workflow('data/multitask')
sentences = ['dog chases cat', 'cat chases dog', 'cat chased by dog', 'an old man on a bench']
projected = workflow.project(sentences)
```
You can then, for example, see how similar the sentences are:

```python
from scipy.spatial.distance import cdist
print 1-cdist(projected, projected, metric='cosine')
```

Train
-----

TODO
