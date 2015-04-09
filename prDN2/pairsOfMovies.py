from __future__ import division
import Orange, sys, time
import matplotlib.pyplot as plt
import numpy as np
import itertools as it


def sort_lists_by(lists, key_list=0, desc=False):
    return zip(*sorted(lists, reverse=desc, key=lambda x: x[key_list]))

data_ = Orange.data.Table("data/data.tab")
item_ = Orange.data.Table("data/item.tab")
data, b, c = data_.to_numpy()
item, b, c = item_.to_numpy()

combination = list(it.combinations(range(np.shape(item)[0]), 2))

pairs_first = []
pairs_second = []

start = time.clock()
avg_mean = []
index1 = []
index2 = []
for comb in combination:
    rez = np.intersect1d(np.nonzero(data[comb[0], :]), np.nonzero(data[comb[1], :]))
    if len(rez) >= 20:
        index1.append(comb[0])
        index2.append(comb[1])
        first_mean = np.mean(data[comb[0], rez])
        second_mean = np.mean(data[comb[1], rez])
        avg_mean.append((first_mean + second_mean)/2)
        pairs_first.append(first_mean)
        pairs_second.append(second_mean)
    else:
        continue

#sort
avg_mean, pairs_first, pairs_second, index1, index2 = [list(x) for x in zip(*sorted(zip(avg_mean, pairs_first, pairs_second, index1, index2), key=lambda pair: pair[0]))]

for i in range(len(avg_mean) - 5, len(avg_mean)):# pairs_first[-5], pairs_second[-5]):
    print avg_mean[i], data_.domain.attributes[index1[i]].name, pairs_first[i], data_.domain.attributes[index2[i]].name, pairs_second[i]

plt.plot(pairs_first, pairs_second, "x")
plt.xlabel("film 1")
plt.ylabel("film 2")
end = time.clock()
plt.show()
#sys.stdout = open("out.txt", "w")
print
print (end - start)/60