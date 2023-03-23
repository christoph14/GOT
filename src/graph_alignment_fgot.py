import os
import argparse
import sqlite3

import networkx as nx
import numpy as np

from fGOT.test_generator_helpers import er_generator, permutation_generator
from utils import check_soft_assignment, check_permutation_matrix
from utils.strategies import get_strategy
from utils.loss_functions import w2_loss, l2_loss


# ArgumentParser
parser = argparse.ArgumentParser(description='Evaluates graph alignment algorithms.')
parser.add_argument('strategies', type=str, nargs='+', help='the strategies to be performed')
parser.add_argument('seed', type=int, help='the used random seed')
# Experiment parameters
parser.add_argument('--within_probability', type=float, default=0.7)
parser.add_argument('--between_probability', type=float, default=0.1)
parser.add_argument('--graph_size', type=int, default=100)
# GOT parameters
parser.add_argument('--alpha', type=float, default=0.1, help='the regularization factor')
parser.add_argument('--it', type=int, default=10, help='number of Sinkhorn iterations')
parser.add_argument('--tau', type=float, default=5, help='the Sinkhorn parameter')
parser.add_argument('--sampling_size', type=int, default=30, help='the sampling size')
parser.add_argument('--iterations', type=int, default=3000, help='the number of iterations')
parser.add_argument('--lr', type=float, default=0.2, help='the learning rate')
# General parameters
parser.add_argument('--path', type=str, default='../results/', help='the path to store the output files')
parser.add_argument('--ignore_log', action='store_const', const=True, default=False, help='disables the log')
parser.add_argument('--allow_soft_assignment', action='store_const', const=True,
                    default=False, help='allow soft assignment instead of a permutation matrix')
args = parser.parse_args()

# Get strategies
strategy_names = args.strategies
strategies = [get_strategy(name, it=args.it, tau=args.tau, n_samples=args.sampling_size, epochs=args.iterations,
                           lr=args.lr, alpha=args.alpha, ones=True, verbose=False) for name in strategy_names]

if not args.ignore_log:
    print('Algorithms:', args.strategies)
    print('Seed:', args.seed)
    print('Path:', args.path)

# Create dictionaries
data = []

# Create random generator
rng = np.random.default_rng(seed=args.seed)

p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for p in p_values:
    n = int(args.graph_size * p)
    L2 = nx.laplacian_matrix(er_generator(n, rng)).todense()
    P_true = permutation_generator(n, seed=args.seed)
    L1 = P_true @ nx.laplacian_matrix((er_generator(n, rng))).todense() @ P_true.T

    # Calculate permutation and different losses for every strategy
    for strategy, name in zip(strategies, strategy_names):
        # Find permutation
        P_estimated = strategy(L1, L2)
        if args.allow_soft_assignment:
            P_estimated = check_soft_assignment(P_estimated, atol=1e-02)
        else:
            P_estimated = check_permutation_matrix(P_estimated, atol=1e-02)

        # Calculate and save different loss functions
        w2_error = w2_loss(L1, L2, P_estimated, alpha=args.alpha)
        l2_error = l2_loss(L1, L2, P_estimated)
        data.append(
            {'strategy' : name,
             'seed' : args.seed,
             'p' : p,
             'w2_loss' : w2_error,
             'l2_loss' : l2_error,
            }
        )
    if not args.ignore_log:
        print(f'p = {p:.2f} done.')

# Save results in database
os.makedirs(args.path, exist_ok=True)
con = sqlite3.connect(f'{args.path}/results_fgot.db', timeout=60)
cur = con.cursor()
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
cur.close()
con.close()
