from __future__ import division
import Orange, sys
import matplotlib.pyplot as plt
import  numpy as np


data = Orange.data.Table("data/data.tab")
item_ = Orange.data.Table("data/item.tab")
# domains = data.domain.name
matrix, b, c = data.to_numpy()
item, b, c = item_.to_numpy()
#item = item[:, 1:]
sys.stdout = open("output.txt", "w")
print matrix.shape
mean = np.average(matrix, axis=0, weights=matrix.astype(bool))[np.newaxis]
print mean.shape
# print item.shape
# print mean.shape, mean.T.shape
mean_per_genre = mean.T * item #np.dot(mean , item) #np.multiply(mean,item.T )
# print mean_per_genre.shape

#masked = mean_per_genre[np.nonzero(mean_per_genre[:,11]),11]
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
# print masked
# print masked.sort(key= lambda obj: obj.mean(), reverse=True)

plt.figure()
plt.boxplot(masked, showmeans=True)
plt.gca().set_xticklabels([i for i in domains], rotation="vertical")
plt.ylabel("Rating")
plt.ylim([0.0, 5.0])
plt.grid(True)
plt.show()

# line
#plt.figure()
val = []
bin = []
for sth, zanr in zip(masked, domains):
    vals, bins = np.histogram(sth.T, bins=5)
    #plt.hist(sth.T)
    #plt.show()
    sum_ = sum(vals)
    vvv = []
    for i in vals:
        i = i/sum_
        vvv.append(i)
        #print i
    #bar plot
    fig = plt.figure()
    plt.bar(bins[1:], vvv, width=bins[1]-bins[0])
    # plt.ylim([0.0, 1.0])
    # plt.xlim([0.0, 5.0])
    # plt.xlabel("rating")
    # plt.ylabel(zanr)
    # plt.grid(True)
    # fname = "dist/" + zanr + "_bar.png"
    # plt.savefig(fname)
    # plt.close(fig)
    # line plot
    # fig = plt.figure()
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

# alternative
# mean = (val[1] + val[2] * 2 + val[3] * 3 + val[4] * 4 + val[5] * 5)/(val[1] + val[2] + val[3] + val[4] + val[5])
# print(val)
# print(mean)
# for i in range(matrix.shape[1]):
#     val = np.bincount(matrix[:,0].astype(int))#  weights=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
#     print val
