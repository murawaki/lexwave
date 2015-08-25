# -*- coding: utf-8 -*-
import sys
from sklearn.decomposition import PCA
import numpy as np

from parse_tree import Node

def extract_mat(root, k):
    # if full == False: only leaves are extracted
    mat_orig = {}
    stack = [root]
    size = -1
    while len(stack) > 0:
        node = stack.pop(0)
        vect = map(lambda x: int(x), node.annotation["&" + k][2:-1])
        size = len(vect)
        mat_orig[node._id] = vect
        if hasattr(node, "left"):
            stack.append(node.left)
            stack.append(node.right)
    mat = np.empty((len(mat_orig), size), dtype=np.int32)
    for _id, vect in mat_orig.iteritems():
        mat[_id] = vect
    return mat

def extract_mat_leaves(root, k):
    # if full == False: only leaves are extracted
    mat_orig = {}
    id2idx = {}
    stack = [root]
    size = -1
    while len(stack) > 0:
        node = stack.pop(0)
        if hasattr(node, "left"):
            stack.append(node.left)
            stack.append(node.right)
        else:
            vect = map(lambda x: int(x), node.annotation["&" + k][2:-1])
            size = len(vect)
            idx = id2idx[node._id] = len(id2idx)
            mat_orig[idx] = vect
    mat = np.empty((len(mat_orig), size), dtype=np.int32)
    for idx, vect in mat_orig.iteritems():
        mat[idx] = vect
    return mat, id2idx

def do_pca(X):
    pca = PCA()
    pca = PCA()
    U, S, V = pca._fit(X)
    X_transformed = np.dot(X - pca.mean_, pca.components_.T)
    return pca, X_transformed

def do_pca_new(pca, X):
    return np.dot(X - pca.mean_, pca.components_.T)

def plot_rec(node, X_transformed, plt, p1, p2):
    _id = node._id
    if hasattr(node, "parent"): # non-root
        _id2 = node.parent._id
        x1, x2 = X_transformed[_id2, p1], X_transformed[_id, p1]
        y1, y2 = X_transformed[_id2, p2], X_transformed[_id, p2]
        if min(abs(x1 - x2), abs(y1 - y2)) > 0.1:
            length_includes_head=True
        else:
            length_includes_head=False
        plt.arrow(x1, y1, x2 - x1, y2 - y1, fc="k", ec="k",
                  length_includes_head=length_includes_head)
        # )
        # head_width=0.05, head_length=0.1 )
        # plt.annotate("", xy=(x2, y2), xytext=(0, 0),
        #              arrowprops=dict(arrowstyle="->"))
    if hasattr(node, "left"):
        plot_rec(node.left, X_transformed, plt, p1, p2)
        plot_rec(node.right, X_transformed, plt, p1, p2)
    if hasattr(node, "left"):
        # internal
        plt.scatter(X_transformed[_id, p1], X_transformed[_id, p2], c="r", s=30)
        if not hasattr(node, "parent"): # root
            plt.annotate("ROOT", (X_transformed[_id, p1], X_transformed[_id, p2]))
    else:
        # leaf
        x, y = X_transformed[_id, p1], X_transformed[_id, p2]
        plt.scatter(x, y, c="g", s=120)
        plt.annotate(node.name, (x, y),
                     xytext=(x + 0.10, y + 0.05))

def main():
    # usage: input key [output]
    #   key: japanese, Ainu_UCLD_GRRW_SDollo, Koreanic_CovUCLD
    import cPickle as pickle
    root = pickle.load(open(sys.argv[1]))
    X = extract_mat(root, sys.argv[2])
    pca, X_transformed = do_pca(X)

    import matplotlib.pyplot as plt
    p1, p2 = 0, 1  # first and second PCs (zero-based numbering)

    plt.figure()
    plt.xlabel("PC%d (%2.1f%%)" % (p1 + 1, pca.explained_variance_ratio_[p1] * 100))
    plt.ylabel("PC%d (%2.1f%%)" % (p2 + 1, pca.explained_variance_ratio_[p2] * 100))
    plot_rec(root, X_transformed, plt, p1, p2)
    plt.legend()
    plt.title('PCA')
    if len(sys.argv) > 3:
        plt.savefig(sys.argv[3], format="png", transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
