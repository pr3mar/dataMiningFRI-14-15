from __future__ import division
import Orange, sys
import matplotlib.pyplot as plt
import  numpy as np

data_ = Orange.data.Table("data/data.tab")
item_ = Orange.data.Table("data/item.tab")

data, b, c = data_.to_numpy()
item, b, c = item_.to_numpy()

views = []
ratings = []

maxView = - (sys.maxint - 1)
indexMaxView = 0
maxViewRating  = 0

maxRating = - (sys.maxint - 1)
indexMaxRating = 0
maxRatingViews = 0

minView = sys.maxint
indexMinView = 0
minViewRating  = 0

minRating = sys.maxint
indexMinRating = 0
minRatingViews = 0

for i in range(np.shape(data)[1]):
    nonzero = np.nonzero(data[:,i])
    views.append(np.shape(nonzero)[1])
    ratings.append(np.mean(data[nonzero,i]))
    if (maxView < views[i]):
        maxView = views[i]
        indexMaxView = i
        maxViewRating = ratings[i]
    if (maxRating < ratings[i]):
        maxRating = ratings[i]
        indexMaxRating = i
        maxRatingViews = views[i]
    if (minView > views[i]):
        minView = views[i]
        indexMinView = i
        minViewRating = ratings[i]
    if (minRating > ratings[i]):
        minRating = ratings[i]
        indexMinRating = i
        minRatingViews = views[i]

print " max views: ", maxViewRating, maxView, data_.domain.attributes[indexMaxView].name, indexMaxView
print " min views: ", minViewRating, minView, data_.domain.attributes[indexMinView].name, indexMinView
print "max rating: ", maxRating, maxRatingViews, data_.domain.attributes[indexMaxRating].name, indexMaxRating
print "min rating: ", minRating, minRatingViews, data_.domain.attributes[indexMinRating].name, indexMinRating

plt.plot(views, ratings, "yo")
plt.xlabel("views")
plt.ylabel("rating")
plt.show()