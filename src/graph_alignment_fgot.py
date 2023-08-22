import os
import argparse
import sqlite3
from time import time

import networkx as nx
import numpy as np
from pygmtools import hungarian

from fGOT.test_generator_helpers import er_generator, permutation_generator
from utils import check_permutation_matrix
from utils.strategies import get_strategy, get_filters
from utils.loss_functions import w2_loss, l2_loss


# ArgumentParser
parser = argparse.ArgumentParser(description='Evaluates graph alignment algorithms.')
parser.add_argument('strategies', type=str, nargs='+', help='the strategies to be performed')
parser.add_argument('seed', type=int, help='the used random seed')
# Experiment parameters
parser.add_argument('--within_probability', type=float, default=0.7)
parser.add_argument('--between_probability', type=float, default=0.1)
parser.add_argument('--graph_size', type=int, default=100)
parser.add_argument('--add_noise', action='store_const', const=True, default=False, help='variable graph size')
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

# Get strategies
strategy_names = args.strategies
strategies = [get_strategy(name, it=args.it, tau=args.tau, n_samples=args.sampling_size, epochs=args.iterations,
                           lr=args.lr, alpha=args.alpha, ones=True, verbose=False, filter_name=args.filter,
                           epsilon=args.epsilon, scale=True) for name in strategy_names]

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
    noise1, noise2 = rng.binomial(4, 0.5, size=2) - 2
    n = int(args.graph_size * p)
    if args.add_noise:
        n1, n2 = n + noise1, n + noise2
    else:
        n1, n2 = n, n
    L2 = nx.laplacian_matrix(er_generator(n1, rng)).todense()
    P_true = permutation_generator(n2, seed=args.seed)
    L1 = P_true @ nx.laplacian_matrix((er_generator(n2, rng))).todense() @ P_true.T

    # Calculate permutation and different losses for every strategy
    for strategy, name in zip(strategies, strategy_names):
        # Find alignment
        start_time = time()
        P_estimated = strategy(L1, L2)
        P_estimated = np.nan_to_num(P_estimated, nan=0.0)
        running_time = time() - start_time
        if args.allow_soft_assignment:
            # P_estimated = check_soft_assignment(P_estimated, atol=1e-02)
            pass
        else:
            P_estimated = check_permutation_matrix(P_estimated, atol=1e-02)

        # Calculate and save different loss functions
        w2_error = w2_loss(L1, L2, P_estimated, alpha=args.alpha)
        l2_error = l2_loss(L1, L2, P_estimated)
        gL1 = get_filters(L1, method="got")
        gL2 = get_filters(L2, method="got")
        approx_error = np.trace(gL1 @ gL1) + np.trace(gL2 @ gL2) - 2 * np.trace(gL1 @ P_estimated.T @ gL2 @ P_estimated)
        data.append(
            {'strategy': name,
             'filter': args.filter,
             'seed': args.seed,
             'p': p,
             'approx_loss': approx_error,
             'w2_loss': w2_error,
             'l2_loss': l2_error,
             'time': running_time,
            }
        )
    if not args.ignore_log:
        print(f'p = {p:.2f} done.')

# Save results in database
os.makedirs(args.path, exist_ok=True)
con = sqlite3.connect(f'{args.path}/results_fgot.db', timeout=60)
cur = con.cursor()

if args.add_noise:
    table_name = "alignment_noise"
else:
    table_name = "alignment"
try:
    cur.execute(f'''CREATE TABLE {table_name} (
                       STRATEGY TEXT NOT NULL,
                       FILTER TEXT NOT NULL,
                       SEED INT NOT NULL,
                       P REAL NOT NULL,
                       W2_LOSS REAL,
                       APPROX_LOSS REAL,
                       L2_LOSS REAL,
                       TIME REAL,
                       unique (STRATEGY, FILTER, SEED, P)
                   )''')
except sqlite3.OperationalError:
    pass

cur.executemany(f"INSERT INTO {table_name} VALUES(:strategy, :filter, :seed, :p, :w2_loss, :approx_loss, :l2_loss, :time)"
                " ON CONFLICT DO UPDATE SET w2_loss=excluded.w2_loss, approx_loss=excluded.approx_loss, l2_loss=excluded.l2_loss, time=excluded.time",
                data)
con.commit()
cur.close()
con.close()
