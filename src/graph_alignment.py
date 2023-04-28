import argparse
import os
import sqlite3

import networkx as nx
import numpy as np

from utils import check_soft_assignment, check_permutation_matrix
from utils.help_functions import remove_edges, graph_from_laplacian
from utils.loss_functions import w2_loss, l2_loss, l2_inv_loss, gw_loss
from utils.strategies import get_strategy

# ArgumentParser
parser = argparse.ArgumentParser(description='Evaluates graph alignment algorithms.')
parser.add_argument('strategies', type=str, nargs='+', help='the strategies to be performed')
parser.add_argument('seed', type=int, help='the used random seed')
# Experiment parameters
parser.add_argument('--within_probability', type=float, default=0.7)
parser.add_argument('--between_probability', type=float, default=0.1)
parser.add_argument('--graph_size', type=int, default=40)
# GOT parameters
parser.add_argument('--alpha', type=float, default=0.1, help='the regularization factor')
parser.add_argument('--it', type=int, default=10, help='number of Sinkhorn iterations')
parser.add_argument('--tau', type=float, default=5, help='the Sinkhorn parameter')
parser.add_argument('--sampling_size', type=int, default=30, help='the sampling size')
parser.add_argument('--iterations', type=int, default=3000, help='the number of iterations')
parser.add_argument('--lr', type=float, default=0.2, help='the learning rate')
# fGOT parameters
parser.add_argument('--filter', type=str)
parser.add_argument('--epsilon', type=float, default=5e-3)
# General parameters
parser.add_argument('--path', type=str, default='../results/', help='the path to store the output files')
parser.add_argument('--ignore_log', action='store_const', const=True, default=False, help='disables the log')
parser.add_argument('--allow_soft_assignment', action='store_const', const=True,
                    default=False, help='allow soft assignment instead of a permutation matrix')
args = parser.parse_args()

# Create results folder
os.makedirs(args.path, exist_ok=True)

# Get strategies
strategy_names = args.strategies
strategies = [get_strategy(name, it=args.it, tau=args.tau, n_samples=args.sampling_size, epochs=args.iterations,
                           lr=args.lr, alpha=args.alpha, filter_name=args.filter, epsilon=args.epsilon, scale=True,
                           ones=True, verbose=False) for name in strategy_names]

if not args.ignore_log:
    print('Algorithms:', args.strategies)
    print('Seed:', args.seed)
    print('Path:', args.path)

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
data = []

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
    G_reduced = remove_edges(G1, communities, between_probability=p, within_probability=0.5, seed=args.seed)
    L_reduced = nx.laplacian_matrix(G_reduced, range(n)).todense()
    L2 = P_true @ L_reduced @ P_true.T
    G2 = graph_from_laplacian(L2)

    # Calculate permutation and different losses for every strategy
    for strategy, name in zip(strategies, strategy_names):
        # Find permutation
        P_estimated = strategy(L1, L2)
        if args.allow_soft_assignment:
            P_estimated = check_soft_assignment(P_estimated, atol=1e-02)
        else:
            P_estimated = check_permutation_matrix(P_estimated, atol=1e-02)

        # Calculate and save different loss functions
        G_aligned = graph_from_laplacian(P_estimated.T @ L2 @ P_estimated)
        w2_error = w2_loss(L1, L2, P_estimated)
        l2_error = l2_loss(L1, L2, P_estimated)
        l2_inv_error = l2_inv_loss(L1, L2, P_estimated, args.alpha, ones=True)
        gw_error = gw_loss(G1, G2, P_estimated.T / n)
        w2_errors[name].append(w2_error)
        l2_errors[name].append(l2_error)
        l2_inv_errors[name].append(l2_inv_error)
        gw_errors[name].append(gw_error)
        data.append(
            {'strategy' : name,
             'seed' : args.seed,
             'p' : p,
             'w2_loss' : w2_error,
             'l2_loss' : l2_error,
             'gw_loss' : gw_error}
        )
    if not args.ignore_log:
        print(f'p = {p:.2f} done.')

# Save results in database
con = sqlite3.connect(f'{args.path}/results_got.db')
cur = con.cursor()
try:
    cur.execute('''CREATE TABLE alignment (
                       STRATEGY TEXT NOT NULL,
                       SEED TEXT NOT NULL,
                       P REAL NOT NULL,
                       W2_LOSS REAL,
                       L2_LOSS REAL,
                       GW_LOSS REAL,
                       unique (STRATEGY, SEED, P)
                   )''')
except sqlite3.OperationalError:
    pass

cur.executemany("INSERT INTO alignment VALUES(:strategy, :seed, :p, :w2_loss, :l2_loss, :gw_loss)"
                " ON CONFLICT DO UPDATE SET w2_loss=excluded.w2_loss, l2_loss=excluded.l2_loss, gw_loss=excluded.gw_loss", data)
con.commit()
cur.close()
con.close()

# os.makedirs(args.path, exist_ok=True)
# for name in strategy_names:
#     np.savetxt(f'{args.path}/l2_inv_error_{name}#{args.seed}.csv', l2_inv_errors[name])
#     np.savetxt(f'{args.path}/w2_error_{name}#{args.seed}.csv', w2_errors[name])
#     np.savetxt(f'{args.path}/l2_error_{name}#{args.seed}.csv', l2_errors[name])
#     np.savetxt(f'{args.path}/gw_error_{name}#{args.seed}.csv', gw_errors[name])
