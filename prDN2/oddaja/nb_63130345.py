from __future__ import division
import Orange, sys
import matplotlib.pyplot as plt
import  numpy as np

data_ = Orange.data.Table("data/data.tab")
cast_ = Orange.data.Table("data/cast.tab")

data, b, c = data_.to_numpy()
cast, b, c = cast_.to_numpy()

avg_by_movie = np.average(data, axis=0, weights=data.astype(bool))[np.newaxis]

avg_by_actor = []
index_of_actor = []

for i in range(cast.shape[1]):
    tmp = avg_by_movie.dot((cast[:,i]).T)
    count = np.shape(np.nonzero(cast[:, i]))[1]
    avg_by_actor.append(tmp/count)
    index_of_actor.append(i)


avg_by_actor = np.array(avg_by_actor).T.tolist()[0]
avg_by_actor, index_of_actor = [list(x) for x in zip(*sorted(zip(avg_by_actor, index_of_actor ), key=lambda pair: pair[0]))]

for i in range(len(avg_by_actor) - 5, len(avg_by_actor)):
    print cast_.domain.attributes[index_of_actor[i]].name,
    for k in avg_by_movie[:,np.nonzero(cast[:, index_of_actor[i]])][0]:
        for j in k:
            print("& {0:.2f}".format(round(j,2))),
    print "\\\\"

vals, bins = np.histogram(avg_by_actor, bins=10)
vvv = []
sum_ = sum(vals)
for i in vals:
    i = i/sum_
    vvv.append(i)
bins = [b0+(b1-b0)/2 for b0, b1 in zip(bins, bins[1:])]
plt.plot(bins, vvv, linewidth=3.0)
plt.xlabel("average rate of cast member")
plt.ylabel("p(cast member)")
plt.show()
