import os
import argparse

import networkx as nx
import numpy as np

from utils.strategies import get_strategy
from utils.help_functions import remove_edges
from utils.loss_functions import w2_loss, l2_loss, l2_inv_loss


# ArgumentParser
parser = argparse.ArgumentParser(description='Evaluates graph alignment algorithms.')
parser.add_argument('strategies', type=str, nargs='+', help='the strategies to be performed')
parser.add_argument('seed', type=int, help='the used random seed')
parser.add_argument('--alpha', dest='alpha', type=float, default=0.1, help='prior for xpal')
parser.add_argument('--it', dest='it', type=int, default=10, help='number of Sinkhorn iterations')
parser.add_argument('--tau', dest='tau', type=float, default=5, help='the Sinkhorn parameter')
parser.add_argument('--sampling_size', dest='sampling_size', type=int, default=30, help='the sampling size')
parser.add_argument('--iterations', dest='iterations', type=int, default=3000, help='the number of iterations')
parser.add_argument('--lr', dest='lr', type=float, default=0.2, help='the learning rate')
parser.add_argument('--regularize', dest='regularize', action='store_const', const=True, default=False, help='regularize laplacian')
parser.add_argument('--path', dest='path', type=str, default='../results/', help='the path to store the output files')
parser.add_argument('--ignore_log', dest='ignore_log', action='store_const', const=True, default=False, help='disables the log')
args = parser.parse_args()

# Get strategies
strategy_names = args.strategies
strategies = [get_strategy(name, it=args.it, tau=args.tau, n_samples=args.sampling_size, epochs=args.iterations,
                           lr=args.lr, alpha=0.1, ones=args.regularize, verbose=False) for name in strategy_names]

# Set parameters for block stochastic model
n = 40
block_size = int(n/4)
blocks = [block_size, block_size, block_size, block_size]
probs = [[0.70, 0.05, 0.05, 0.05],
         [0.05, 0.70, 0.05, 0.05],
         [0.05, 0.05, 0.70, 0.05],
         [0.05, 0.05, 0.05, 0.70]]

# Create dictionaries
w2_errors = {}
l2_errors = {}
l2_inv_errors = {}
for name in strategy_names:
    w2_errors[name] = []
    l2_errors[name] = []
    l2_inv_errors[name] = []

# Create random generator
rng = np.random.default_rng(seed=args.seed)

# Generate original graph
G1 = nx.stochastic_block_model(blocks, probs, seed=args.seed)
assert nx.is_connected(G1), 'G1 is not connected.'
communities = {}
for node in G1.nodes:
    communities[node] = np.floor(node / block_size)

n = len(G1)
L1 = nx.laplacian_matrix(G1, range(n))
L1 = np.double(np.array(L1.todense()))

# Generate permutation matrix
idx = rng.permutation(n)
P_true = np.eye(n)
P_true = P_true[idx, :]

p_values = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
for p in p_values:
    # Randomly remove edges from the original graph and permute it
    G1_reduced = remove_edges(G1, communities, between_probability=p, within_probability=0.5, seed=rng)
    L1_reduced = nx.laplacian_matrix(G1_reduced, range(n))
    L1_reduced = np.double(np.array(L1_reduced.todense()))
    L2 = P_true @ L1_reduced @ P_true.T

    # Calculate permutation and different losses for every strategy
    for strategy, name in zip(strategies, strategy_names):
        # Find permutation
        P_estimated = strategy(L1, L2)

        # Calculate and save different loss functions
        w2_error = w2_loss(L1, L2, P_estimated, args.alpha, ones=args.regularize)
        l2_error = l2_loss(L1, L2, P_estimated)
        l2_inv_error = l2_inv_loss(L1, L2, P_estimated, args.alpha, ones=args.regularize)
        w2_errors[name].append(w2_error)
        l2_errors[name].append(l2_error)
        l2_inv_errors[name].append(l2_inv_error)
        np.savetxt(f'{args.path}/permutation_{name}_{p}#{args.seed}.csv', l2_inv_errors[name])
    if not args.ignore_log:
        print(f'p = {p:.2f} done.')

# Save results
os.makedirs(args.path, exist_ok=True)
for name in strategy_names:
    np.savetxt(f'{args.path}/l2_inv_error_{name}#{args.seed}.csv', l2_inv_errors[name])
    np.savetxt(f'{args.path}/w2_error_{name}#{args.seed}.csv', w2_errors[name])
    np.savetxt(f'{args.path}/l2_error_{name}#{args.seed}.csv', l2_errors[name])
