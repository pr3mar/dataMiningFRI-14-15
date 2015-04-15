try:
    import sys
    sys.path.remove(filter(lambda e: "orange3" in e, sys.path)[0])
except:
    pass

import Orange
import numpy as np

### Use as desired:
import Orange
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from itertools import combinations


def count(data, value):
    c = 0
    for i in data:
        if(data[i] == value):
            c += 1
    return c

# Load data
Utab = Orange.data.Table("../data/user.tab")
Itab = Orange.data.Table("../data/item.tab")
Dtab = Orange.data.Table("../data/data.tab")
Ctab = Orange.data.Table("../data/cast.tab")

U, _, _ = Utab.to_numpy()
I, _, _ = Itab.to_numpy()
D, _, _ = Dtab.to_numpy()
C, _, _ = Ctab.to_numpy()

# Movies per genre
print Itab.domain

MPG = I.sum(axis=0).astype(int)
print "Movies per genre:", MPG


# Select genres with at least T movies
T = 50
inxs = map(int, np.where(I.sum(axis=0) >= T)[0])
Itab.domain = [Itab.domain[i] for i in inxs]
I = I[:, inxs]
MPG = MPG[inxs]
print "Filtered genres", Itab.domain
for i in range(len(Itab.domain)):
    print i, Itab.domain[i].name


###########
#  PART 1 #
###########


# Prepare data and impute
# VPG: Views per genre
# UG: User genre rating
# Also, Do not account for averages with less than 5 samples
VPG = (D > 0).dot(I)
UG = np.nan_to_num(D.dot(I) / VPG)

for r, c in zip(*((VPG < 5).nonzero())):
    inxs = (VPG[:, c] >= 5).nonzero()[0]
    UG[r, c] = np.mean(UG[inxs, c])

# Normalize UG to the average of each users respective rates
UG = UG / UG.mean(axis=1).reshape((len(UG), 1))

# YOUR CODE HERE

# 1.a code
k = 4
model = KMeans(n_clusters=k)
y = model.fit_predict(UG)
appearance = np.bincount(y)
# question 1.a answer
print appearance

# 1.b code
cluster_indices = []
for i in range(k):
    cluster_indices.append(np.where(y == i)[0])
avg_by_genre = UG.mean(axis=0)

avg_per_cluster = []
for i in range(k):
    avg_per_cluster.append(UG[cluster_indices[i],:].mean(axis=0))
avg_all = []
for i in avg_per_cluster:
    avg_all.append(i/avg_by_genre)

# print the max values
# question 1.b answer
max_genre = []
for i in avg_all:
    max_genre.append({i.argmin():max(i)})

print max_genre

# 1.c code and answer
combination = list(combinations(range(k), 2))
color = {0:"b", 1:"y", 2:"r", 3:"g" }
colors = []
for i in y:
    colors.append(color[i])
for i in combination:
    movieX = max_genre[i[0]].keys()[0]
    movieY = max_genre[i[1]].keys()[0]
    plt.figure()
    plt.xlabel(Itab.domain[movieX].name)
    plt.ylabel(Itab.domain[movieY].name)
    plt.scatter(UG[:, movieX], UG[:, movieY], c=colors, s=50.0, edgecolors='none')
    plt.show()

# question 1.d


# ###########
# #  PART 2 #
# ###########
#
# # Select movies with at least 250 views
# # Their indices are in inxs
# T = 250
# inxs = map(int, np.where((D > 0).sum(axis=0) > T)[0])
# n = len(inxs)
# print "Number of movies: ", n
#
# # YOUR CODE HERE





