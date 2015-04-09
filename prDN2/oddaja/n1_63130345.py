from __future__ import division
import Orange, sys
import matplotlib.pyplot as plt
import  numpy as np


data = Orange.data.Table("data/data.tab")
item_ = Orange.data.Table("data/item.tab")

matrix, b, c = data.to_numpy()
item, b, c = item_.to_numpy()

print matrix.shape
mean = np.average(matrix, axis=0, weights=matrix.astype(bool))[np.newaxis]
print mean.shape
mean_per_genre = mean.T * item #np.dot(mean , item) #np.multiply(mean,item.T )
masked = []
for i in range(18):
    masked.append(mean_per_genre[np.nonzero(mean_per_genre[:,i]),i])
domains = []
for i in item_.domain.attributes:
    domains.append(i.name)
means = []
for i in masked:
    means.append(np.mean(i))
means, masked, domains = [list(x) for x in zip(*sorted(zip(means, masked, domains), key=lambda pair: pair[0]))]

plt.figure()
plt.boxplot(masked, showmeans=True)
plt.gca().set_xticklabels([i for i in domains], rotation="vertical")
plt.ylabel("Rating")
plt.ylim([0.0, 5.0])
plt.grid(True)
plt.show()

val = []
bin = []
for sth, zanr in zip(masked, domains):
    vals, bins = np.histogram(sth.T, bins=5)
    sum_ = sum(vals)
    vvv = []
    for i in vals:
        i = i/sum_
        vvv.append(i)
    fig = plt.figure()
    plt.bar(bins[1:], vvv, width=bins[1]-bins[0])
    bins = [(b0 + (b1-b0)/2) + bins[1] - bins[0] for b0, b1 in zip(bins, bins[1:])]
    plt.plot(bins, vvv, "r", linewidth=3.0)
    plt.ylabel(zanr)
    plt.xlabel("rating")
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 5.0])
    plt.grid(True)
    fname = "dist/" + zanr + "_nn.png"
    plt.savefig(fname)
    plt.close(fig)