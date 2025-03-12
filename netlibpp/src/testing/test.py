from scipy.spatial.distance import pdist, squareform
from itertools import permutations, combinations
from math import comb
import numpy as np

from sklearn.neighbors import radius_neighbors_graph
import networkx as nx

import sys
sys.path.insert(0, '/home/kurk/curse/filtration/graph_func.cpython-312-x86_64-linux-gnu.so')

import graph_func as gf
from timeit import default_timer as timer

print(gf.__file__)
print(np.__file__)
print(gf.filtrate.__doc__)

n = 30
X = np.random.normal(size=(n, 2))
A = squareform(pdist(X))

#############################################

num_cores = 1

start = timer()
es = gf.filtrate(A, 2, [1, 2], num_cores)
ts = gf.filtrate(A, 3, [1, 2], num_cores)
qs = gf.filtrate(A, 4, [1, 2], num_cores)
end = timer()
print("c++ ver time on {} threads: {}".format(num_cores, end - start))
#############################################

num_cores = 2

start = timer()
es = gf.filtrate(A, 2, [1, 2], num_cores)
ts = gf.filtrate(A, 3, [1, 2], num_cores)
qs = gf.filtrate(A, 4, [1, 2], num_cores)
end = timer()
print("c++ ver time on {} threads: {}".format(num_cores, end - start))

###########################################