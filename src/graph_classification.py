import sys
import argparse

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, zero_one_loss
from sklearn.neighbors import KNeighborsClassifier

from utils.distances import wasserstein_distance, gw_distance
from utils.help_functions import random_permutation, graph_from_laplacian
from utils.strategies import get_strategy

# ArgumentParser
parser = argparse.ArgumentParser(description='Evaluates graph classification algorithms.')
parser.add_argument('strategy', type=str, help='the strategy to be performed')
args = parser.parse_args()

graphs = []
permuted_graphs = []
permutation_matrices = []
y = []

n = 20
graphs_per_class = 20

# Stochastic Block Model with 2 blocks (SBM2)
sizes = [10, 10]
p = [[0.75, 0.15],
     [0.15, 0.75]]
for _ in range(graphs_per_class):
    G = nx.stochastic_block_model(sizes, p)
    L = np.double(np.array(nx.laplacian_matrix(G, range(n)).todense()))
    P = random_permutation(n)
    graphs.append(L)
    permuted_graphs.append(P @ L @ P.T)
    permutation_matrices.append(P)
    y.append(0)

# Stochastic Block Model with 3 blocks (SBM3)
sizes = [8, 8, 4]
p = [[0.80, 0.25, 0.25],
     [0.25, 0.80, 0.25],
     [0.25, 0.25, 0.80]]
for _ in range(graphs_per_class):
    G = nx.stochastic_block_model(sizes, p)
    L = np.double(np.array(nx.laplacian_matrix(G, range(n)).todense()))
    P = random_permutation(n)
    graphs.append(L)
    permuted_graphs.append(P @ L @ P.T)
    permutation_matrices.append(P)
    y.append(1)

# random regular graph (RG)
for _ in range(graphs_per_class):
    # Exactly (n/2) * degree edges
    # 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, ...
    degree = 8
    G = nx.random_regular_graph(degree, n)
    L = np.double(np.array(nx.laplacian_matrix(G, range(n)).todense()))
    P = random_permutation(n)
    graphs.append(L)
    permuted_graphs.append(P @ L @ P.T)
    permutation_matrices.append(P)
    y.append(2)

# Barabasy-Albert model (BA)
for _ in range(graphs_per_class):
    # Exactly (n-m) * m edges
    # 19, 36, 51, 64, 75, 84, 91, 96, 99, 100, 99, 96, ...
    G = nx.barabasi_albert_graph(n, 6)
    L = np.double(np.array(nx.laplacian_matrix(G, range(n)).todense()))
    P = random_permutation(n)
    graphs.append(L)
    permuted_graphs.append(P @ L @ P.T)
    permutation_matrices.append(P)
    y.append(3)

# Watts-Strogatz model (WS)
for _ in range(graphs_per_class):
    # Exactly (n/2) * k edges, k even
    # 20, 40, 60, 80, 100, ...
    G = nx.watts_strogatz_graph(n, k=8, p=0.2)
    L = np.double(np.array(nx.laplacian_matrix(G, range(n)).todense()))
    P = random_permutation(n)
    graphs.append(L)
    permuted_graphs.append(P @ L @ P.T)
    permutation_matrices.append(P)
    y.append(4)

strategy = get_strategy(args.strategy, it=10, tau=5, n_samples=30, epochs=1500,
                        lr=0.2, alpha=0.1, ones=True, verbose=False)
alignment_matrices = []
for idx, (L, L_permuted) in enumerate(zip(graphs, permuted_graphs)):
    alignment_matrices.append(strategy(L, L_permuted))
    sys.stdout.write(f'\r{idx+1} graphs done.')

aligned_graphs = [P.T @ L @ P for L, P in zip(permuted_graphs, alignment_matrices)]

if args.strategy.lower() in ['got', 'fgot']:
    distances = [[wasserstein_distance(L, L_aligned) for L in graphs] for L_aligned in aligned_graphs]
elif args.strategy.lower() in ['gw']:
    distances = [[gw_distance(graph_from_laplacian(L), graph_from_laplacian(L_aligned)) for L in graphs] for L_aligned in aligned_graphs]
else:
    distances = [[np.linalg.norm(L - L_aligned, ord='fro') for L in graphs] for L_aligned in aligned_graphs]

y = np.array(y)
knn = KNeighborsClassifier(n_neighbors=1, metric='precomputed').fit(distances, y)
indices = knn.kneighbors(distances, return_distance=False)
indices = indices[:, 0]
y_pred = y[indices]
print()
print(confusion_matrix(y, y_pred))
print('Correct classifications:', np.trace(confusion_matrix(y, y_pred)))
ConfusionMatrixDisplay.from_predictions(y, y_pred, colorbar=False)
plt.title(f'{args.strategy}: {100-zero_one_loss(y, y_pred, normalize=False)}/100')
plt.savefig(f'../plots/confusion_matrix_{args.strategy}.pdf', bbox_inches='tight')
plt.show()
