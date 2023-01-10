import sys

import networkx as nx
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from utils.distances import wasserstein_distance
from utils.help_functions import random_permutation
from utils.strategies import get_strategy

graphs = []
permuted_graphs = []
permutation_matrices = []
y = []

n = 20
graphs_per_class = 10

# Stochastic Block Model with 2 blocks (SBM2)
sizes = [10, 10]
p = [[0.65, 0.1],
     [0.1, 0.65]]
for _ in range(graphs_per_class):
    G = nx.stochastic_block_model(sizes, p)
    # print(G.number_of_edges())
    L = np.double(np.array(nx.laplacian_matrix(G, range(n)).todense()))
    P = random_permutation(n)
    graphs.append(L)
    permuted_graphs.append(P @ L @ P.T)
    permutation_matrices.append(P)
    y.append(0)

# Stochastic Block Model with 3 blocks (SBM3)
sizes = [8, 8, 4]
p = [[0.8, 0.17, 0.17],
     [0.17, 0.8, 0.17],
     [0.17, 0.17, 0.8]]
for _ in range(graphs_per_class):
    G = nx.stochastic_block_model(sizes, p)
    # print(G.number_of_edges())
    L = np.double(np.array(nx.laplacian_matrix(G, range(n)).todense()))
    P = random_permutation(n)
    graphs.append(L)
    permuted_graphs.append(P @ L @ P.T)
    permutation_matrices.append(P)
    y.append(1)

# random regular graph (RG)
for _ in range(graphs_per_class):
    G = nx.random_regular_graph(7, n)
    # print(G.number_of_edges())
    L = np.double(np.array(nx.laplacian_matrix(G, range(n)).todense()))
    P = random_permutation(n)
    graphs.append(L)
    permuted_graphs.append(P @ L @ P.T)
    permutation_matrices.append(P)
    y.append(2)

# Barabasy-Albert model (BA)
for _ in range(graphs_per_class):
    G = nx.barabasi_albert_graph(n, 5)
    L = np.double(np.array(nx.laplacian_matrix(G, range(n)).todense()))
    P = random_permutation(n)
    graphs.append(L)
    permuted_graphs.append(P @ L @ P.T)
    permutation_matrices.append(P)
    y.append(3)

# Watts-Strogatz model (WS)
for _ in range(graphs_per_class):
    G = nx.watts_strogatz_graph(n, k=8, p=0.2)
    L = np.double(np.array(nx.laplacian_matrix(G, range(n)).todense()))
    P = random_permutation(n)
    graphs.append(L)
    permuted_graphs.append(P @ L @ P.T)
    permutation_matrices.append(P)
    y.append(4)

strategy = get_strategy('L2', it=10, tau=5, n_samples=30, epochs=1500,
                        lr=0.2, alpha=0.1, ones=True, verbose=False)
alignment_matrices = []
for idx, (L, L_permuted) in enumerate(zip(graphs, permuted_graphs)):
    alignment_matrices.append(strategy(L, L_permuted))
    sys.stdout.write(f'\r{idx+1} graphs done.')

aligned_graphs = [P.T @ L @ P for L, P in zip(permuted_graphs, alignment_matrices)]
distances = [[wasserstein_distance(L, L_aligned) for L in graphs] for L_aligned in aligned_graphs]

y = np.array(y)
knn = KNeighborsClassifier(n_neighbors=1, metric='precomputed').fit(distances, y)
indices = knn.kneighbors(distances, return_distance=False)
indices = indices[:, 0]
y_pred = y[indices]
print()
print(confusion_matrix(y, y_pred))
