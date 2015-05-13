import numpy as np
import sys
from misc import dotp

def nmf(Xs, rank, max_iter=100, allow_negative=False):
    """
    Standard non-negative matrix factorization.
    :param Xs
        Data matrices.
    :param rank
        Factorization rank (r).
    :param max_iter.
        Maximum number of iterations.
    :param getobj
        Return vector of cost function values.
    :return:
        W: A common row clustering matrix.
        Hs: A list of data-source specific matrices.
    """
    m, Ns = Xs[0].shape[0], [x.shape[1] for x in Xs]
    W = np.random.rand(m, rank)
    Hs = [np.random.rand(rank, n) for n in Ns]

    for itr in range(max_iter):
        sys.stderr.write("%d/%d\r" % (itr, max_iter) )
        sys.stderr.flush()

        enum = dotp([Xs[0], Hs[0].T])
        denom = dotp([W, dotp([Hs[0], Hs[0].T])])
        for i in range(1, len(Hs)):
            enum += dotp([Xs[i], Hs[i].T])
            denom += dotp([W, dotp([Hs[i], Hs[i].T])])
        W = np.nan_to_num(W * enum/denom)

        for i in range(len(Hs)):
            enum = dotp([W.T, Xs[i]])
            denom = dotp([W.T, W, Hs[i]])
            Hs[i] = np.nan_to_num(Hs[i] * enum/denom)

    return W, Hs


def nmf_fix(Xs, Hs, rank, max_iter=100):
    """
    :param Xs:
        A list of matrices with descriptions of test examples.
    :param Hs:
        A list of matrices for a pre-built model.
    :param rank:
        Factorization rank
    :param max_iter:
        Max. number of iterations.
    :return:
        W: A predicted row clustering matrix.
    """
    m, Ns = Xs[0].shape[0], [x.shape[1] for x in Xs]
    W = np.random.rand(m, rank)

    for itr in range(max_iter):
        enum = dotp([Xs[0], Hs[0].T])
        denom = dotp([W, dotp([Hs[0], Hs[0].T])])
        for i in range(1, len(Xs)):
            enum += dotp([Xs[i], Hs[i].T])
            denom = dotp([W, dotp([Hs[i], Hs[i].T])])
        W = np.nan_to_num(W * enum/denom)

    return W


