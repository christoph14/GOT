import os
import argparse
import sqlite3

import networkx as nx
import numpy as np

from fGOT.test_generator_helpers import er_generator, permutation_generator
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
parser.add_argument('--graph_size', dest='graph_size', type=int, default=100)
# GOT parameters
parser.add_argument('--alpha', dest='alpha', type=float, default=0.1, help='the regularization factor')
parser.add_argument('--it', dest='it', type=int, default=10, help='number of Sinkhorn iterations')
parser.add_argument('--tau', dest='tau', type=float, default=5, help='the Sinkhorn parameter')
parser.add_argument('--sampling_size', dest='sampling_size', type=int, default=30, help='the sampling size')
parser.add_argument('--iterations', dest='iterations', type=int, default=3000, help='the number of iterations')
parser.add_argument('--lr', dest='lr', type=float, default=0.2, help='the learning rate')
# General parameters
parser.add_argument('--path', dest='path', type=str, default='../results/', help='the path to store the output files')
parser.add_argument('--ignore_log', dest='ignore_log', action='store_const', const=True, default=False, help='disables the log')
parser.add_argument('--allow_soft_assignment', dest='allow_soft_assignment', action='store_const', const=True,
                    default=False, help='allow soft assignment instead of a permutation matrix')
parser.add_argument('--reset_results', action='store_const', const=True, default=False, help='delete old results')
args = parser.parse_args()

# Create results folder
os.makedirs(args.path, exist_ok=True)

# Get strategies
strategy_names = args.strategies
strategies = [get_strategy(name, it=args.it, tau=args.tau, n_samples=args.sampling_size, epochs=args.iterations,
                           lr=args.lr, alpha=args.alpha, ones=True, verbose=False) for name in strategy_names]

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
# G1 = nx.stochastic_block_model(blocks, probs, seed=rng)
# L1 = nx.laplacian_matrix(G1, range(n)).todense()
# assert nx.is_connected(G1), 'G1 is not connected.'
# communities = {}
# for node in G1.nodes:
#     communities[node] = np.floor(node / block_size)

p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for p in p_values:
    n = int(args.graph_size * p)
    L2 = nx.laplacian_matrix(er_generator(n)).todense()
    P_true = permutation_generator(n)
    L1 = P_true @ nx.laplacian_matrix((er_generator(n))).todense() @ P_true.T

    # Calculate permutation and different losses for every strategy
    for strategy, name in zip(strategies, strategy_names):
        # Find permutation
        P_estimated = strategy(L1, L2)
        if args.allow_soft_assignment:
            P_estimated = check_soft_assignment(P_estimated, atol=1e-02)
        else:
            P_estimated = check_permutation_matrix(P_estimated, atol=1e-02)

        # Calculate and save different loss functions
        # G_aligned = graph_from_laplacian(P_estimated.T @ L2 @ P_estimated)
        w2_error = w2_loss(L1, L2, P_estimated)
        l2_error = l2_loss(L1, L2, P_estimated)
        # l2_inv_error = l2_inv_loss(L1, L2, P_estimated, args.alpha, ones=True)
        # gw_error = gw_distance(G1, G_aligned)
        w2_errors[name].append(w2_error)
        l2_errors[name].append(l2_error)
        # l2_inv_errors[name].append(l2_inv_error)
        # gw_errors[name].append(gw_error)
        data.append(
            {'strategy' : name,
             'seed' : args.seed,
             'p' : p,
             'w2_loss' : w2_error,
             'l2_loss' : l2_error,
             # 'gw_loss' : gw_error
            }
        )
    if not args.ignore_log:
        print(f'p = {p:.2f} done.')

# Save results in database
con = sqlite3.connect(f'{args.path}/fgot_results.db', timeout=100)
cur = con.cursor()
if args.reset_results:
    cur.execute('''DROP TABLE alignment''')
try:
    cur.execute('''CREATE TABLE alignment (
                       STRATEGY TEXT NOT NULL,
                       SEED TEXT NOT NULL,
                       P REAL NOT NULL,
                       W2_LOSS REAL,
                       L2_LOSS REAL,
                       unique (STRATEGY, SEED, P)
                   )''')
except sqlite3.OperationalError:
    pass

cur.executemany("INSERT INTO alignment VALUES(:strategy, :seed, :p, :w2_loss, :l2_loss)"
                " ON CONFLICT DO UPDATE SET w2_loss=excluded.w2_loss, l2_loss=excluded.l2_loss", data)
con.commit()

# os.makedirs(args.path, exist_ok=True)
# for name in strategy_names:
#     np.savetxt(f'{args.path}/l2_inv_error_{name}#{args.seed}.csv', l2_inv_errors[name])
#     np.savetxt(f'{args.path}/w2_error_{name}#{args.seed}.csv', w2_errors[name])
#     np.savetxt(f'{args.path}/l2_error_{name}#{args.seed}.csv', l2_errors[name])
#     np.savetxt(f'{args.path}/gw_error_{name}#{args.seed}.csv', gw_errors[name])
