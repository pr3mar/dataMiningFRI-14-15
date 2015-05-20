from itertools import combinations, product
import numpy as np
import sys


def pgdnmf(X, rank, max_iter=100, tol=1e-5, eta=1e-2):
    """
        NMF using projected gradient descent.
        Minimize || X - WH ||_F, subject to W, H >= 0.


        :param X:
            Data matrix.
        :param rank
            Maximal rank of the model (W, H).
        :param max_iter
            Maximum number of iterations.

        :return
            Data model W, H.
    """
    m, n  = X.shape
    W     = np.random.rand(m, rank)
    H     = np.random.rand(rank, n)
    itr   = 0
    error = float("inf")

    known = X.nonzero()

    while error > tol and itr < max_iter:
        error = 0
        if itr % 10 == 0:
            sys.stdout.write("%s/%s\r" % (itr, max_iter))
            sys.stdout.flush()

        dw = np.zeros((m, rank))
        dh = np.zeros((rank, n))
        for i, j in zip(*known):
            for k in xrange(rank):
                dw[i, k]    = dw[i, k] + (X[i, j] - W[i, :].dot(H[:, j])) * H[k, j]
                dh[k, j]    = dh[k, j] + (X[i, j] - W[i, :].dot(H[:, j])) * W[i, k]       
            error += (X[i, j] - W[i, :].dot(H[:, j]))**2

        W = W + eta * dw
        H = H + eta * dh
        W = W * (W > 0)
        H = H * (H > 0)
        itr += 1

    return W, H


if __name__ == "__main__":
    G = np.random.rand(10, 3)
    X = G.dot(G.T)
    # rang (rank) je sicer pomemben parameter, ki vpliva na 
    # kvaliteto oz. natancnost faktorizacije
    # za namene DN uporabite rank=2
    W, H = pgdnmf(X, rank=2, max_iter=100)
    print "Residual error: ", np.linalg.norm(X - W.dot(H), ord="fro")**2





