try:
    # Only important if you have orange3 also
    import sys
    sys.path.remove(filter(lambda e: "orange3" in e, sys.path)[0])
except:
    pass


# Use as desired
import Orange
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score


# Load data tables
Utab = Orange.data.Table("../data/user.tab")
Itab = Orange.data.Table("../data/item.tab")
Dtab = Orange.data.Table("../data/data.tab")
Ctab = Orange.data.Table("../data/cast.tab")
U, _, _ = Utab.to_numpy()
I, _, _ = Itab.to_numpy()
D, _, _ = Dtab.to_numpy()
C, _, _ = Ctab.to_numpy()
print "U", U.shape
print "I", I.shape
print "D", D.shape
print "C", C.shape



# Implementation of a simple majority classifier for instructive purposes
class Majority:

    majority = None

    def __init__(self, ):
        pass

    def fit(self, X, y):
        """
            Fit classifier on training data.
        """
        self.majority = 1 if np.sum(y == 1) >= np.sum(y == 0) else 0

    def predict(self, Xtest):
        """
            Predict test data.
        """
        return np.ones((len(Xtest), )) * self.majority



# Sample code: PART 1
# Prediction of whether the user likes a movie or not (class), based on all other users
#   All other users are now considered as 'attributes' of the classification problem.
i = 302     # Randomly choose a user on index 302
X = np.hstack([D[:i, :].T, D[i+1:, :].T])   # Attributes are all other users
y = np.hstack([D[i, :]])  # Class labels are movie ratings of user 301
index_set = np.where(y > 0)[0]  # Select only the movies that user watched
X = X[index_set, :]     # Training data - attributes
y = y[index_set]     # Training data - class labels

# Set threshold to convert rating to two classes;
# Ratings > tau are considered as "Like" (class 1) and otherwise as "Unlike" (class 0)
tau = 2
y = y > tau

# Sample usage of models
model = Majority()
model.fit(X, y)                 # Train model on all data
prediction = model.predict(X)   # Predict the same training data
P = precision_score(y, prediction)   # Measure precision score
print "Precision of the majority vote: %f" % P




#######################
# Sample code: PART 2 #
#######################
# Classification of users; separation between occupations.
# How well can we separate eg. scientists and salesmen based on movie ratings?
# Build classifier for each pair of jobs
# Exclude 'none' and 'other'
jobs = sorted(list(set([str(u["occupation"]) for u in Utab])))
jobs.remove("none")
jobs.remove("other")

j1 = "scientist"
j2 = "salesman"
inxs1 = map(lambda e: e[0], filter(lambda e: str(e[1]["occupation"]) == j1, enumerate(Utab)))
inxs2 = map(lambda e: e[0], filter(lambda e: str(e[1]["occupation"]) == j2, enumerate(Utab)))
X = D[inxs1+inxs2, :]                                # Select appropriate indices
y = np.array(len(inxs1) * [0] + len(inxs2) * [1])    # Assign users to two classes

# Select top 200 viewed movies for the pair of users' types
top_movie_inxs = np.argsort((X > 0).sum(axis=0))[::-1][:200]
X = X[:, top_movie_inxs]


# Train a decision tree and export tree graph in .dot format
# To draw, install
#       http://www.graphviz.org/
#   and run:
#       dot -Tpdf jobs.dot -o graph.pdf
model = DecisionTreeClassifier()
model.fit(X, y)
with open("jobs.dot", 'w') as f:
    export_graphviz(model, out_file=f, feature_names=map(str, [Dtab.domain.attributes[mi].name for mi in top_movie_inxs]))




