import sys
import argparse

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, zero_one_loss

from utils.distances import wasserstein_distance, gw_distance
from utils.help_functions import random_permutation
from utils.strategies import get_strategy

# ArgumentParser
parser = argparse.ArgumentParser(description='Evaluates graph classification algorithms.')
parser.add_argument('strategy', type=str, help='the strategy to be performed')
parser.add_argument('--seed', type=int, help='the random seed')
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

graphs = []
permuted_graphs = []
permutation_matrices = []
y = []

n = 20
graphs_per_class = 20
n_graphs = 5 * graphs_per_class

# Stochastic Block Model with 2 blocks (SBM2)
sizes = [10, 10]
p = [[0.75, 0.15],
     [0.15, 0.75]]
for _ in range(graphs_per_class):
    G = nx.stochastic_block_model(sizes, p, seed=rng)
    while not nx.is_connected(G):
        G = nx.stochastic_block_model(sizes, p, seed=rng)
    L = np.double(np.array(nx.laplacian_matrix(G, range(n)).todense()))
    P = random_permutation(n, rng)
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
    G = nx.stochastic_block_model(sizes, p, seed=rng)
    while not nx.is_connected(G):
        G = nx.stochastic_block_model(sizes, p, seed=rng)
    L = np.double(np.array(nx.laplacian_matrix(G, range(n)).todense()))
    P = random_permutation(n, rng)
    graphs.append(L)
    permuted_graphs.append(P @ L @ P.T)
    permutation_matrices.append(P)
    y.append(1)

# random regular graph (RG)
for _ in range(graphs_per_class):
    # Exactly (n/2) * degree edges
    # 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, ...
    degree = 8
    G = nx.random_regular_graph(degree, n, seed=rng)
    while not nx.is_connected(G):
        G = nx.random_regular_graph(degree, n, seed=rng)
    L = np.double(np.array(nx.laplacian_matrix(G, range(n)).todense()))
    P = random_permutation(n, rng)
    graphs.append(L)
    permuted_graphs.append(P @ L @ P.T)
    permutation_matrices.append(P)
    y.append(2)

# Barabasy-Albert model (BA)
for _ in range(graphs_per_class):
    # Exactly (n-m) * m edges
    # 19, 36, 51, 64, 75, 84, 91, 96, 99, 100, 99, 96, ...
    m = 5
    G = nx.barabasi_albert_graph(n, m, seed=rng)
    while not nx.is_connected(G):
        G = nx.barabasi_albert_graph(n, m, seed=rng)
    L = np.double(np.array(nx.laplacian_matrix(G, range(n)).todense()))
    P = random_permutation(n, rng)
    graphs.append(L)
    permuted_graphs.append(P @ L @ P.T)
    permutation_matrices.append(P)
    y.append(3)

# Watts-Strogatz model (WS)
for _ in range(graphs_per_class):
    # Exactly (n/2) * k edges, k even
    # 20, 40, 60, 80, 100, ...
    k = 8
    G = nx.watts_strogatz_graph(n, k, p=0.5, seed=rng)
    while not nx.is_connected(G):
        G = nx.watts_strogatz_graph(n, k, p=0.5, seed=rng)
    L = np.double(np.array(nx.laplacian_matrix(G, range(n)).todense()))
    P = random_permutation(n, rng)
    graphs.append(L)
    permuted_graphs.append(P @ L @ P.T)
    permutation_matrices.append(P)
    y.append(4)

strategy = get_strategy(args.strategy, it=10, tau=5, n_samples=30, epochs=1500,
                        lr=0.2, alpha=0.1, ones=True, verbose=False)
distances = np.full((len(graphs), len(graphs)), np.inf)
for i, L1 in enumerate(permuted_graphs):
    for j, L2 in enumerate(permuted_graphs):
        if i == j: continue
        P = strategy(L1, L2)
        L_aligned = P.T @ L2 @ P
        if args.strategy.lower() in ['got', 'fgot']:
            distances[i, j] = wasserstein_distance(L1, L_aligned)
        # elif args.strategy.lower() in ['gw']:
        #     distances[i,j] = gw_distance(graph_from_laplacian(L1), graph_from_laplacian(L_aligned))
        else:
            distances[i,j] = np.linalg.norm(L1 - L_aligned, ord='fro')
    sys.stdout.write(f'\r{i + 1} graphs done')

y = np.array(y)
nearest_neighbors = np.nanargmin(distances, axis=0)
y_pred = y[nearest_neighbors]
accuracy = len(graphs) - zero_one_loss(y, y_pred, normalize=False)

print()
print(confusion_matrix(y, y_pred))
print('Correct classifications:', accuracy)
ConfusionMatrixDisplay.from_predictions(y, y_pred, colorbar=False)
plt.title(f'{args.strategy}: {accuracy}/{len(graphs)}')
plt.savefig(f'../plots/confusion_matrix_{args.strategy}.pdf', bbox_inches='tight')
plt.show()
