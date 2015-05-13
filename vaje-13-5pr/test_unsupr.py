import sys  
try:
    sys.path.remove(filter(lambda e: "orange3" in e, sys.path)[0])
except:
    pass

import Orange
import numpy as np
import matplotlib.pyplot as plt
from nmf import nmf

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

# Unsupervised learning of training examples
W, Hs = nmf([X], rank=rank)
Xa = W.dot(Hs[0])
print Xa.shape

plt.figure(figsize=(5.0, 5.0))
plt.imshow(X)
plt.title("Original")

plt.figure(figsize=(5.0, 5.0))
plt.title("Approximation")
plt.imshow(Xa)

# Show clusters
if show_features:
    plt.figure()
    plt.title("Clusters")
    plt.pcolor(W)
    plt.figure()
    plt.title("Features")
    plt.pcolor(Hs[0])


# Show all figures
plt.show()

