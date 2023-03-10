import os
import argparse

import networkx as nx
import numpy as np

from utils import check_soft_assignment, check_permutation_matrix
from utils.distances import gw_distance
from utils.strategies import get_strategy
from utils.help_functions import remove_edges, graph_from_laplacian
from utils.loss_functions import w2_loss, l2_loss, l2_inv_loss


# ArgumentParser
parser = argparse.ArgumentParser(description='Evaluates graph alignment algorithms.')
parser.add_argument('strategies', type=str, nargs='+', help='the strategies to be performed')
parser.add_argument('seed', type=int, help='the used random seed')
# Experiment parameters
parser.add_argument('--within_probability', dest='within_probability', type=float, default=0.7)
parser.add_argument('--between_probability', dest='between_probability', type=float, default=0.1)
parser.add_argument('--graph_size', dest='graph_size', type=int, default=40)
# GOT parameters
parser.add_argument('--alpha', dest='alpha', type=float, default=0.1, help='the regularization factor')
parser.add_argument('--it', dest='it', type=int, default=10, help='number of Sinkhorn iterations')
parser.add_argument('--tau', dest='tau', type=float, default=5, help='the Sinkhorn parameter')
parser.add_argument('--sampling_size', dest='sampling_size', type=int, default=30, help='the sampling size')
parser.add_argument('--iterations', dest='iterations', type=int, default=3000, help='the number of iterations')
parser.add_argument('--lr', dest='lr', type=float, default=0.2, help='the learning rate')
parser.add_argument('--path', dest='path', type=str, default='../results/', help='the path to store the output files')
parser.add_argument('--ignore_log', dest='ignore_log', action='store_const', const=True, default=False, help='disables the log')
args = parser.parse_args()

# Create results folder
os.makedirs(args.path, exist_ok=True)

# Get strategies
strategy_names = args.strategies
strategies = [get_strategy(name, it=args.it, tau=args.tau, n_samples=args.sampling_size, epochs=args.iterations,
                           lr=args.lr, alpha=0.1, ones=True, verbose=False) for name in strategy_names]

# Set parameters for block stochastic model
n = args.graph_size
p = args.within_probability
q = args.between_probability
n_blocks = 4
block_size = int(n/n_blocks)
blocks = [block_size] * n_blocks
probs = [[p, q, q, q],
         [q, p, q, q],
         [q, q, p, q],
         [q, q, q, p]]

# Create dictionaries
w2_errors = {}
l2_errors = {}
l2_inv_errors = {}
gw_errors ={}
for name in strategy_names:
    w2_errors[name] = []
    l2_errors[name] = []
    l2_inv_errors[name] = []
    gw_errors[name] = []

# Create random generator
rng = np.random.default_rng(seed=args.seed)

# Generate original graph
G1 = nx.stochastic_block_model(blocks, probs, seed=rng)
L1 = nx.laplacian_matrix(G1, range(n)).todense()
assert nx.is_connected(G1), 'G1 is not connected.'
communities = {}
for node in G1.nodes:
    communities[node] = np.floor(node / block_size)

# Generate permutation matrix
idx = rng.permutation(n)
P_true = np.eye(n)
P_true = P_true[idx, :]

p_values = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
for p in p_values:
    G_reduced = remove_edges(G1, communities, between_probability=p, within_probability=0.5, seed=rng)
    L_reduced = nx.laplacian_matrix(G_reduced, range(n)).todense()
    L2 = P_true @ L_reduced @ P_true.T

    # Calculate permutation and different losses for every strategy
    for strategy, name in zip(strategies, strategy_names):
        # Find permutation
        P_estimated = strategy(L1, L2)
        try:
            P_estimated = check_soft_assignment(P_estimated, atol=0.01)
        except ValueError as e:
            print(f'Error in seed {args.seed}')
            print('Col sums:', P_estimated.sum(axis=0))
            print('Row sums:', P_estimated.sum(axis=1))
            raise e
        P_estimated = check_permutation_matrix(P_estimated)

        # Calculate and save different loss functions
        G_aligned = graph_from_laplacian(P_estimated.T @ L2 @ P_estimated)
        w2_error = w2_loss(L1, L2, P_estimated, args.alpha, ones=args.regularize)
        l2_error = l2_loss(L1, L2, P_estimated)
        l2_inv_error = l2_inv_loss(L1, L2, P_estimated, args.alpha, ones=args.regularize)
        gw_error = gw_distance(G1, G_aligned)
        w2_errors[name].append(w2_error)
        l2_errors[name].append(l2_error)
        l2_inv_errors[name].append(l2_inv_error)
        gw_errors[name].append(gw_error)
        np.savetxt(f'{args.path}/permutation_{name}_{p}#{args.seed}.csv', P_estimated)
    if not args.ignore_log:
        print(f'p = {p:.2f} done.')

# Save results
os.makedirs(args.path, exist_ok=True)
for name in strategy_names:
    np.savetxt(f'{args.path}/l2_inv_error_{name}#{args.seed}.csv', l2_inv_errors[name])
    np.savetxt(f'{args.path}/w2_error_{name}#{args.seed}.csv', w2_errors[name])
    np.savetxt(f'{args.path}/l2_error_{name}#{args.seed}.csv', l2_errors[name])
    np.savetxt(f'{args.path}/gw_error_{name}#{args.seed}.csv', gw_errors[name])
