try:
    # Only important if you have orange3 also
    import sys
    sys.path.remove(filter(lambda e: "orange3" in e, sys.path)[0])
except:
    pass


# Use as desired
import Orange, time
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score


Utab = Orange.data.Table("../data/user.tab")
Itab = Orange.data.Table("../data/item.tab")
Dtab = Orange.data.Table("../data/data.tab")
Ctab = Orange.data.Table("../data/cast.tab")
U, _, _ = Utab.to_numpy()
I, _, _ = Itab.to_numpy()
D, _, _ = Dtab.to_numpy()
C, _, _ = Ctab.to_numpy()

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

def printData(data, file):
    sys.stdout = open(file, "w")
    for i in data:
        for j in i:
            # for k in j:
            print j,
        print
    sys.stdout = sys.__stdout__
def getClasses():
    tau = [2, 3, 4]
    for t in tau:
        avgs = [
            [[], [], []],
            [[], [], []],
            [[], [], []],
            [[], [], []]
        ]
        user = 0
        for i in range(U.shape[0]):
            X = np.hstack([D[:i, :].T, D[i+1:, :].T])   # Attributes are all other users
            y = np.hstack([D[i, :]])  # Class labels are movie ratings of user 301
            index_set = np.where(y > 0)[0]  # Select only the movies that user watched
            X = X[index_set, :]     # Training data - attributes
            y = y[index_set]     # Training data - class labels
            print user
            try:
                split = StratifiedShuffleSplit(y, test_size=0.1, train_size=0.9)
                y = y > t
                majority_precision = 0
                majority_recall = 0
                majority_area = 0

                bayes_precision = 0
                bayes_recall = 0
                bayes_area = 0

                kn_precision = 0
                kn_recall = 0
                kn_area = 0

                dt_precision = 0
                dt_recall = 0
                dt_area = 0
                j = 0
                for train_indice, test_indice in split:
                    j += 1
                    train_set = X[train_indice, :]
                    train_set_labels = y[train_indice]
                    test_set = X[test_indice, :]
                    test_set_labels = y[test_indice]

                    model = Majority()
                    model.fit(train_set, train_set_labels)
                    prediction = model.predict(test_set)
                    majority_precision += precision_score(test_set_labels, prediction)
                    majority_recall += recall_score(test_set_labels, prediction)
                    majority_area += roc_auc_score(test_set_labels, prediction)

                    model = GaussianNB()
                    prediction = model.fit(train_set, train_set_labels).predict(test_set)
                    bayes_precision += precision_score(test_set_labels, prediction)
                    bayes_recall += recall_score(test_set_labels, prediction)
                    bayes_area += roc_auc_score(test_set_labels, prediction)

                    model = KNeighborsClassifier()
                    model.fit(train_set, train_set_labels)
                    prediction = model.predict(test_set)
                    kn_precision += precision_score(test_set_labels, prediction)
                    kn_recall += recall_score(test_set_labels, prediction)
                    kn_area += roc_auc_score(test_set_labels, prediction)

                    model = DecisionTreeClassifier()
                    prediction = model.fit(train_set, train_set_labels).predict(test_set)
                    dt_precision += precision_score(test_set_labels, prediction)
                    dt_recall += recall_score(test_set_labels, prediction)
                    dt_area += roc_auc_score(test_set_labels, prediction)
                avgs[0][0].append(majority_precision/j)
                avgs[0][1].append(majority_recall/j)
                avgs[0][2].append(majority_area/j)

                avgs[1][0].append(bayes_precision/j)
                avgs[1][1].append(bayes_recall/j)
                avgs[1][2].append(bayes_area/j)

                avgs[2][0].append(kn_precision/j)
                avgs[2][1].append(kn_recall/j)
                avgs[2][2].append(kn_area/j)

                avgs[3][0].append(dt_precision/j)
                avgs[3][1].append(dt_recall/j)
                avgs[3][2].append(dt_area/j)
            except:
                pass
    for i in avgs:
        print "t = ", t,
        for j in i:
            try:
                print sum(j)/len(j),
            except:
                print 0,
        print

def getNeighbours(t):
    kn_all = []
    neighbors = [1, 2, 3, 5, 10, 15, 30, 60, 90, 100]
    kn_users = np.zeros((3, len(neighbors)))
    kn_users = np.zeros((3, len(neighbors)))
    user = 0

    for i in range(U.shape[0]):
        X = np.hstack([D[:i, :].T, D[i+1:, :].T])   # Attributes are all other users
        y = np.hstack([D[i, :]])  # Class labels are movie ratings of user 301
        index_set = np.where(y > 0)[0]  # Select only the movies that user watched
        X = X[index_set, :]     # Training data - attributes
        y = y[index_set]     # Training data - class labels
        print "[t =", t, "], user = ", user
        try:
            split = StratifiedShuffleSplit(y, test_size=0.1, train_size=0.9)
            y = y > t
            kn_local = np.zeros((3, len(neighbors)))
            j = 0
            for train_indice, test_indice in split:
                j += 1
                train_set = X[train_indice, :]
                train_set_labels = y[train_indice]
                test_set = X[test_indice, :]
                test_set_labels = y[test_indice]
                for l in range(len(neighbors)):
                    kn = KNeighborsClassifier(n_neighbors=neighbors[l])
                    kn.fit(train_set, train_set_labels)
                    prediction = kn.predict(test_set)
                    kn_local[0, l] += precision_score(test_set_labels, prediction)
                    kn_local[1, l] += recall_score(test_set_labels, prediction)
                    try:
                        kn_local[2, l] += roc_auc_score(test_set_labels, prediction)
                    except:
                        kn_local[2, l] = 0
            for q in range(kn_local.shape[0]):
                for k in range(kn_local.shape[1]):
                    kn_users[q, k] += kn_local[q, k]/j
        except:
            user -=1
        user += 1
        # break
    print "[t = ", t, "]altogether = ", user
    for i in range(kn_users.shape[0]):
        for j in range(kn_users.shape[1]):
            kn_users[i, j] = kn_users[i,j]/user
    # kn_all.append(kn_users)
    printData(kn_users, ("data_tau_" + str(t)))
    plt.figure()
    # print i.shape
    plt.title("t = " + str(t))
    plt.plot(np.array(neighbors), np.array(kn_users[0]), label="Precision")
    plt.plot(np.array(neighbors), np.array(kn_users[1]), label="Recall")
    plt.plot(np.array(neighbors), np.array(kn_users[2]), label="ROC")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Load data tables
    print "U", U.shape
    print "I", I.shape
    print "D", D.shape
    print "C", C.shape
    pool = Pool(processes=20)
    start = time.clock()
    # getNeighbours(2)
    pool.map(getNeighbours, range(2, 5))
    # getClasses()
    print time.clock() - start


# #######################
# # Sample code: PART 2 #
# #######################
# # Classification of users; separation between occupations.
# # How well can we separate eg. scientists and salesmen based on movie ratings?
# # Build classifier for each pair of jobs
# # Exclude 'none' and 'other'
# jobs = sorted(list(set([str(u["occupation"]) for u in Utab])))
# jobs.remove("none")
# jobs.remove("other")
#
# j1 = "scientist"
# j2 = "salesman"
# inxs1 = map(lambda e: e[0], filter(lambda e: str(e[1]["occupation"]) == j1, enumerate(Utab)))
# inxs2 = map(lambda e: e[0], filter(lambda e: str(e[1]["occupation"]) == j2, enumerate(Utab)))
# X = D[inxs1+inxs2, :]                                # Select appropriate indices
# y = np.array(len(inxs1) * [0] + len(inxs2) * [1])    # Assign users to two classes
#
# # Select top 200 viewed movies for the pair of users' types
# top_movie_inxs = np.argsort((X > 0).sum(axis=0))[::-1][:200]
# X = X[:, top_movie_inxs]
#
#
# # Train a decision tree and export tree graph in .dot format
# # To draw, install
# #       http://www.graphviz.org/
# #   and run:
# #       dot -Tpdf jobs.dot -o graph.pdf
# model = DecisionTreeClassifier()
# model.fit(X, y)
# with open("jobs.dot", 'w') as f:
#     export_graphviz(model, out_file=f, feature_names=map(str, [Dtab.domain.attributes[mi].name for mi in top_movie_inxs]))




