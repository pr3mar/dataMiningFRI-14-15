from __future__ import division
from pandas.compat import lzip
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
    genreX = Itab.domain[movieX].name
    genreY = Itab.domain[movieY].name
    fig = plt.figure()
    plt.xlabel(genreX)
    plt.ylabel(genreY)
    plt.scatter(UG[:, movieX], UG[:, movieY], c=colors, s=50.0, edgecolors='none')
    fname = "../pics/kmeans_porazdelitev_" + genreX + "_" + genreY + "_63130345.png"
    plt.savefig(fname)
    plt.close(fig)

# question 1.d
# occupation
occupation =  Utab.domain["occupation"]
occupation_per_cluster = []
occupation_data_set = {}
for j in occupation:
    occupation_data_set[j.value] = 0
for i in Utab[:]:
    occupation_data_set[i[2].value] += 1
results_per_sluster = []
results_per_sluster_proff = []
# print "occupation data set:"
# print occupation_data_set

# gender
gender = Utab.domain["gender"]
gender_data_set = {}
for j in gender:
    gender_data_set[j.value] = 0
for i in Utab[:]:
    gender_data_set[i[1].value] += 1

# age
age = Utab.domain["age"]
age_data_set = {}
for i in range(10):
    age_data_set[i] = 0
for i in Utab[:]:
    #print i[0], i[0] // 10,  i[0] % 10
    age_data_set[i[0] // 10] += 1
print "age:"
print age_data_set

for i in range(k):
    print "cluster", i + 1
    # occupation
    cluster = {}
    for j in occupation:
        cluster[j.value] = 0
    for j in cluster_indices[i]:
        cluster[Utab[j][2].value] += 1
    rez = []
    rez_prof = []
    for key, v in cluster.items():
        # print v, occupation_data_set[k]
        rez.append(v/occupation_data_set[key])
        rez_prof.append(key)
    print len(cluster_indices[i])
    occupation_per_cluster.append(cluster)
    sorted_lists = sorted(lzip(rez, rez_prof), reverse=True, key=lambda x: x[0])
    rez, rez_prof = [[x[p] for x in sorted_lists] for p in range(2)]
    for j in range(3):
        print "%.3f %s" % (rez[j], rez_prof[j])
    results_per_sluster.append(rez)
    results_per_sluster_proff.append(rez_prof)

    # gender
    gender_cluster = {}
    for j in gender:
        gender_cluster[j.value] = 0
    for j in cluster_indices[i]:
        gender_cluster[Utab[j][1].value] += 1
    print "gender"
    for v,key in gender_cluster.items():
        print "%s = %.3f, " % (v, key/len(cluster_indices[i]))
    # rez_gender = []
    # rez_gender_ = []
    # for key,v in gender_cluster.items():
    #     rez_gender.append(v/gender_data_set[key])
    #     rez_gender_.append(key)
    # for j in range(2):
    #     print "%.3f %s" % (rez_gender[j], rez_gender_[j])

    #age
    age_cluster = {}
    for j in range(10):
        age_cluster[j] = 0
    for j in cluster_indices[i]:
        # print Utab[j][0], Utab[j][0] // 10 #,  i[0] % 10
        age_cluster[Utab[j][0] // 10] += 1
    print "age", age_cluster
    print

###########
#  PART
# 2 #
###########

# Select movies with at least 250 views
# Their indices are in inxs
T = 250
inxs = map(int, np.where((D > 0).sum(axis=0) > T)[0])
n = len(inxs)
print "Number of movies: ", n

# YOUR CODE HERE
from itertools import product
avg_movies = np.average(D, axis=1, weights=D.astype(bool))
#print avg_movies.shape
#print avg_movies
movies = D[inxs,:].T
zero_vals = np.array(np.where(movies == 0))
print zero_vals.shape

# for i in range(939):
#     print avg_movies[i]
#     for j in range(68):
#         print movies[i,j],
#     print

for i in range(939):
    for j in range(68):
        if movies[i,j] == 0:
            movies[i,j] = avg_movies[i]

# for i in range(939):
#     print avg_movies[i]
#     for j in range(68):
#         print movies[i,j],
#     print

Distances = np.zeros((n, n))
euc = lambda x, y: np.sqrt(np.sum((x-y)**2))
for i, j in product(range(n), range(n)):
    xi = movies[i]
    xj = movies[j]
    Distances[i, j] = euc(xi, xj)

names = []
for inst in inxs:
    names.append(Dtab.domain[inst].name)

L = linkage(Distances, method="average")
plt.figure()
dend = dendrogram(L)
leaves = dend["leaves"]
plt.gca().set_xticklabels([names[i] for i in leaves])
#plt.savefig("../pics/dendrogram_63130345.svg")
plt.show()