import gzip
import imaginet
import cPickle
import utils

import sys

def main():
    inp = sys.argv[1]
    out = sys.argv[2]
    model = cPickle.load(gzip.open(inp))
    cPickle.dump(utils.serialize(model), gzip.open(out, 'w'))

main()
