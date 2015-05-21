from imaginet import *

workflow = load_workflow('data/multitask')

sentences = ['dog chases cat', 'cat chases dog', 'cat chased by dog', 'an old man on a bench']

projected = workflow.project(sentences)

# Check distances
from scipy.spatial.distance import cdist

print cdist(projected, projected, metric='cosine')
