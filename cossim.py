# -*- coding: utf-8 -*-
import sys
from sklearn.decomposition import PCA
import numpy as np

from parse_tree import Node

def create_leaf2vect(root, k):
    leaf2vect = {}
    stack = [root]
    size = -1
    while len(stack) > 0:
        node = stack.pop(0)
        vect = np.array(map(lambda x: int(x), node.annotation['&' + k][2:-1]), dtype=np.int32)
        # vect = map(lambda x: int(x), node.annotation['&Ainu_UCLD_GRRW_SDollo'][2:-1])
        # vect = map(lambda x: int(x), node.annotation['&japanese'][2:-1])
        size = len(vect)
        if hasattr(node, "left"):
            stack.append(node.left)
            stack.append(node.right)
        else:
            leaf2vect[node.name] = vect
    return leaf2vect

def main():
    import cPickle as pickle
    root = pickle.load(open(sys.argv[1]))
    leaf2vect = create_leaf2vect(root, sys.argv[2])
    vect1 = leaf2vect[sys.argv[3]]
    leaf2sim = {}
    for k, vect2 in leaf2vect.iteritems():
        leaf2sim[k] = (vect1 * vect2).sum() / np.sqrt((vect1 * vect1).sum() * (vect2 * vect2).sum())
    sorted_leaves = sorted(leaf2sim.keys(), key=lambda x: leaf2sim[x], reverse=True)
    for k in sorted_leaves:
        sys.stdout.write("%s\t%f\n" % (k, leaf2sim[k]))

if __name__ == "__main__":
    main()
