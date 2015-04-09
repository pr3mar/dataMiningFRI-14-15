from __future__ import division
import Orange, sys, time
import matplotlib.pyplot as plt
import numpy as np
import itertools as it

data_ = Orange.data.Table("new.tab")
domains = data_.domain.attributes
data, x, y = data_.to_numpy()

data = data.T

for i, k in zip(data, domains):
    print k.name,
    for j in i:
        print("& {0:.2f}".format(round(j,2))),
    print "\\\\"