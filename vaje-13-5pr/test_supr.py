import sys  
try:
    sys.path.remove(filter(lambda e: "orange3" in e, sys.path)[0])
except:
    pass

import Orange
import numpy as np
import matplotlib.pyplot as plt
from nmf import nmf, nmf_fix
from sklearn.metrics import roc_auc_score

# Select factorization rank
if len(sys.argv) > 1:
    rank = int(sys.argv[1])
else:
    rank = 3

# Show features
if len(sys.argv) > 2:
    show_features = int(sys.argv[2])
else:
    show_features = 0


# Example data - impute to replace missing values
data = Orange.data.Table("yeast-class-RPR")
imputer = Orange.feature.imputation.MinimalConstructor()
imputer = imputer(data)
data = imputer(data)

# Convert to numpy and separate positive and negative values
# Note that NMF works on strictly *nonnegative data*
X, y, _ = data.to_numpy()
Xp = X * (X > 0)
Xn = X * (X < 0) * (-1)
X = np.hstack([Xp, Xn])

# Reshape and select target class
y = y.reshape((len(X), 1))
y = (y == 0).astype(int)


# Learn the model parameters given the attributes and the class values
tr = range(0, len(X), 2)
te = range(1, len(X), 2)

W, Hs = nmf([X[tr], y[tr]], rank=rank)
Wte  = nmf_fix([X[te]], [Hs[0]], rank=rank)
prediction = Wte.dot(Hs[1])

# Measure performance
auc = roc_auc_score(y_score=prediction, y_true=y[te])
print "AUC: ", auc


