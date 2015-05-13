import scipy.sparse as sp
import numpy as np

def dotp(args):
    """
        Dot product for both sparse and non-sparse matrices.
        :param args
            List of NumPy arrays and/or SciPY sparse matrices.
        :return
            Result of dot product. If any matrix in args was sparse, dot product is sparse.
    """
    if len(args) == 1:
        return args[0]
    indicator = [sp.isspmatrix(t) for t in args]
    if any(indicator):
        i = indicator.index(True)
        j = i+1
        if i <= len(args) - 2:
            # i is not last
            return dotp(args[:i] + [args[i].dot(args[j])] + args[j+1:])
        else:
            # i is last
            j = i - 1
            return dotp(args[:j] + [args[i].T.dot(args[j].T).T])
    else:
        # no matrix is sparse
        # find maximum eliminated dimension (TODO)
        return dotp([args[0].dot(args[1])]+args[2:])


def pattern_matrix(m, n, n_patterns, density=0.7):
    """
        Generate a matrix with patterns.
        :param m
            No. of rows.
        :param n
            No. of columns
        :param n_patterns
            No. of patterns
        :param density
            Desired density of matrix.
        :return
            Generated matrix.
    """
    patts = np.random.rand(n_patterns, n)
    mask = patts > density
    patts *= mask
    M = np.zeros((m, n))
    for mi in range(m):
        M[mi, :] = patts[np.random.randint(0, n_patterns), :]
    return M


def error(Xs, W, Hs, target=None):
    """
    Calculate approximation error.
    :param Xs
        Data matrices.
    :param W
        Clustering matrix.
    :param Hs
         Input Hs matrices.
    """
    if isinstance(target, type(None)):
        return np.sum([np.linalg.norm(x-W.dot(h), ord=2)**2 for x, h in zip(Xs, Hs)])
    else:
        return np.linalg.norm(Xs[target]-W.dot(Hs[target]), ord=2)**2


def density(X):
    """
        Helper function to evaluate density.
        :return
            Matrix density.
    """
    return 1.0 * len(np.array(X > 1e-5).nonzero()[0]) / (np.product(X.shape))


def sparseness(X, axis=None):
    """
        Sparseness as defined by Hoyer, 2004.
        :param X
            Matrix.
        :return
            Sparseness measure
    """
    if sp.isspmatrix(X):
        X = np.array(X.todense())
    no2 = np.linalg.norm(X, ord=2)
    if no2:
        if isinstance(axis, type(None)):
            n = np.product(X.shape)
            return (np.sqrt(n) - np.linalg.norm(X, ord=1)/no2) / (np.sqrt(n) - 1)
        else:
            return np.mean([(np.sqrt(len(w)) - np.linalg.norm(w, ord=1)/np.linalg.norm(w, ord=2)) / (np.sqrt(len(w)) - 1)
                            for w in np.rollaxis(X, axis)])
    else:
        return 1.0
