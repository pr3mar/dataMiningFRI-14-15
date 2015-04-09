from __future__ import division
import Orange, sys, time, math
import matplotlib.pyplot as plt
import  numpy as np
import itertools as it

start = time.clock()
data_ = Orange.data.Table("data/data.tab")
item_ = Orange.data.Table("data/item.tab")

data, b, c = data_.to_numpy()
item, b, c = item_.to_numpy()

avg_by_genre = np.zeros([2,18])

avg_by_usr = []

for j in range(data.shape[0]):
    avg = []
    for i in range(item.shape[1]):
        tmp = data[j,:] * item[:, i]
        count = len(list(np.nonzero(tmp))[0])
        if count == 0:
            rez = -1
        else:
            sum_ = sum(tmp)
            rez = (sum_ / count)
            avg_by_genre[1][i] += 1
            avg_by_genre[0][i] += rez
        avg.append(rez)
    avg_by_usr.append(avg)

avg_by_genre = np.array(avg_by_genre[0] / avg_by_genre[1])


for j in range(len(avg_by_usr)):
    for i in range(len(avg_by_usr[j])):
        if avg_by_usr[j][i] == -1:
            avg_by_usr[j][i] = avg_by_genre[i]
avg_by_usr = np.array(avg_by_usr)

end = time.clock()
print end - start
usrs = [range(len(avg_by_usr))]
sys.stdout = open("find_outliers.tab", "w")
for x in ["user","Action", "Adventure", "Animation", "Childrens", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror", "Musical", "Mystery", "Romance", "Sci_Fi", "Thriller", "War", "Western"]:
    print "%s\t" % x,
print
for x in ["s", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c",]:
    print "%s\t" % x,
print
i = 1
for row in avg_by_usr:
    print "%i\t" % i,
    for val in row:
        print "%f\t" % val,
    print
    i += 1