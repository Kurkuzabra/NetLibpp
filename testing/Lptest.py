from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance_matrix
from itertools import permutations, combinations
from math import comb
import numpy as np
from timeit import default_timer as timer
import sys
sys.path.insert(0, '/home/kurk/curse/filtration/graph_func.cpython-312-x86_64-linux-gnu.so')

import graph_func as gf
import importlib
importlib.reload(gf)
from timeit import default_timer as timer

from sklearn.neighbors import radius_neighbors_graph
import networkx as nx

print(gf.__file__)
print(gf.filtrate.__doc__)

X = np.array([
    [0, 2.75], # 1
    [2, 2], # 2
    [3, 2.25], # 3
    [4, 4], #4
    [3.5, 0.75], #5
    [2.5, -0.25], #6
    [2.25, -2.25], #7
    [2.5, -4], #8
    [1, -3], #9
    [-1, -2.75], #10
    [-2.5, -1.0], #11
    [-4, 0.5], #12
    [-2, 1] #13
])

X = X + np.random.normal(0, 0.05, X.shape)
X

__K = gf.get_Lp_from_coord_matrix(X, 2, 2, 4)
__K.as_list()


# import gudhi

# # Input points
# points = np.array([
#     [ 0.05838514,  2.72923084],
#     [ 2.05108295,  2.08260998],
#     [ 3.02475067,  2.22048891],
#     [ 4.04415356,  3.92873094],
#     [ 3.44521162,  0.70378788],
#     [ 2.45550644, -0.24627632],
#     [ 2.31440411, -2.26545732],
#     [ 2.51532972, -4.10935871],
#     [ 1.20678458, -3.01752814],
#     [-0.94723886, -2.78622397],
#     [-2.50026482, -0.95464206],
#     [-3.96030553,  0.45824035],
#     [-1.8986244 ,  1.03383127]
# ])

# # Create a Vietoris-Rips complex
# max_dimension = 2  # Maximum simplex dimension
# max_filtration = 2  # Maximum filtration threshold
# rips_complex = gudhi.RipsComplex(points=points, max_edge_length=max_filtration)

# # Create the simplex tree
# simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)

# # Print the filtration
# print("Filtration:")
# for simplex, filtration_value in simplex_tree.get_filtration():
#     print(f"Simplex: {simplex}, Filtration Value: {filtration_value}")
