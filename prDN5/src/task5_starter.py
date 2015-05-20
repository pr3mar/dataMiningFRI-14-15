try:
    # Only important if you have orange3 also
    import sys
    sys.path.remove(filter(lambda e: "orange3" in e, sys.path)[0])
except:
    pass


# Use as desired
import Orange
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from pgdnmf import pgdnmf

# Load data tables
Utab = Orange.data.Table("../data/user.tab")
Itab = Orange.data.Table("../data/item.tab")
Dtab = Orange.data.Table("../data/data.tab")
Ctab = Orange.data.Table("../data/cast.tab")
U, _, _ = Utab.to_numpy()
I, _, _ = Itab.to_numpy()
D, _, _ = Dtab.to_numpy()
C, _, _ = Ctab.to_numpy()
print "U", U.shape
print "I", I.shape
print "D", D.shape
print "C", C.shape


# Take first 100 users/items for a start
D = D[:100, :100]


tau     = 3.0                               # like / unlike threshold
ratings = D.nonzero()                       # all pairs of available ratings
classes = D[ratings] > tau                  # class vectors

# Build a factor model from all data
W, H = pgdnmf(D, rank=10, max_iter=1000)
print "Residual error: ", np.linalg.norm(D-W.dot(H), ord="fro")**2


