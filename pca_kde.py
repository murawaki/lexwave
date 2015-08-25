# -*- coding: utf-8 -*-
import sys
from scipy.stats import gaussian_kde
import numpy as np

from parse_tree import TreeParser, label_clades
from pca_tree import extract_mat_leaves, do_pca, do_pca_new

def plot_leaves_rec(node, X_transformed, id2idx, plt, p1, p2):
    _id = node._id
    if hasattr(node, "left"):
        plot_leaves_rec(node.left, X_transformed, id2idx, plt, p1, p2)
        plot_leaves_rec(node.right, X_transformed, id2idx, plt, p1, p2)
    if _id in id2idx:
        # leaf
        idx = id2idx[_id]
        x, y = X_transformed[idx, p1], X_transformed[idx, p2]
        plt.scatter(x, y, c="g", s=120)
        plt.annotate(node.name, (x, y),
                     xytext=(x + 0.10, y + 0.05))

def main():
    # usage: input key [output]
    #   key: &japanese, &Ainu_UCLD_GRRW_SDollo, &Koreanic_CovUCLD
    tp = TreeParser(sys.argv[1])
    k = sys.argv[2]
    tid = int(sys.argv[3])
    clade_name = sys.argv[4]
    burnin = 50
    p1, p2 = 0, 1  # first and second PCs (zero-based numbering)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    clade_vect_list = []
    for i in xrange(burnin, len(tp.n.trees.trees)):
        root = tp.parse(i)
        clade_dict = label_clades(root)
        clade = clade_dict[clade_name]
        vect = map(lambda x: int(x), clade.annotation["&" + k][2:-1])
        clade_vect_list.append(vect)
        if i == tid:
            X, id2idx = extract_mat_leaves(root, k)
            pca, X_transformed = do_pca(X)

            # plt.figure()
            plt.xlabel("PC%d (%2.1f%%)" % (p1 + 1, pca.explained_variance_ratio_[p1] * 100))
            plt.ylabel("PC%d (%2.1f%%)" % (p2 + 1, pca.explained_variance_ratio_[p2] * 100))
            plot_leaves_rec(root, X_transformed, id2idx, ax, p1, p2)

    Y = np.empty((len(clade_vect_list), X.shape[1]), dtype=np.int32)
    for i, vect in enumerate(clade_vect_list):
        Y[i] = vect
    Y_transformed = do_pca_new(pca, Y)

    val = np.vstack((Y_transformed[:,p1], Y_transformed[:,p2]))
    kernel = gaussian_kde(val)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    _X, _Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([_X.ravel(), _Y.ravel()])
    Z = np.reshape(kernel(positions).T, _X.shape)
    plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])

    plt.legend()
    plt.title('PCA')
    if len(sys.argv) > 5:
        plt.savefig(sys.argv[5], format="png", transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
