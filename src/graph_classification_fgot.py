#!/home/we661512/miniconda3/envs/graph/bin/python
#SBATCH --job-name=fgot_distances
#SBATCH --output=../log_slurm/fgot_classification_%A.log
#SBATCH --mem-per-cpu=3G
#SBATCH --time 360
#SBATCH --account=thes1398
#SBATCH --ntasks=1


import argparse
import os
import pickle
import sys

sys.path.append(os.getcwd())

import networkx as nx
import numpy as np

from utils.strategies import get_strategy



if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluates graph alignment algorithms.')
    parser.add_argument('graph1', type=int, help='the first graph for which the distances are computed')
    parser.add_argument('graph2', type=int, help='the second graph for which the distances are computed')
    parser.add_argument('seed', type=int, help='the used random seed')
    parser.add_argument('--path', type=str, default='../results/fgot_distances', help='the path to store the output files')
    args = parser.parse_args()

    # Load graph data set
    n_graphs = 100
    path = "../data/ENZYMES/enzymes.pkl"
    with open(path, 'rb') as file:
        graphs = pickle.load(file)
    rng = np.random.default_rng(seed=args.seed)
    X = np.random.choice(graphs, n_graphs)
    y = np.array([G.graph['label'] for G in X])

    # Compute and save distances
    strategy = get_strategy('fgot', it=10, tau=1, n_samples=30, epochs=1, lr=0.2)
    G1 = X[args.graph1]
    G2 = X[args.graph2]
    L1 = nx.laplacian_matrix(G1).todense()
    L2 = nx.laplacian_matrix(G2).todense()
    P = strategy(L1, L2)
    distance = np.linalg.norm(L1 - P.T @ L2 @ P, ord='fro')
    os.makedirs(args.path, exist_ok=True)
    np.savetxt(f'{args.path}/{args.graph1}-{args.graph2}#{args.seed}.csv', distance)

    # # Compute and save accuracy
    # nearest_neighbors = np.argmin(distances, axis=0)
    # y_pred = y_train[nearest_neighbors]
    # accuracy = n_graphs - zero_one_loss(y_test, y_pred, normalize=False)
    #
    # # Save results in database
    # os.makedirs(args.path, exist_ok=True)
    # con = sqlite3.connect(f'{args.path}/results_fgot.db', timeout=60)
    # cur = con.cursor()
    # try:
    #     cur.execute('''CREATE TABLE classification (
    #                        STRATEGY TEXT NOT NULL,
    #                        DATA TEXT NOT NULL,
    #                        SEED TEXT NOT NULL,
    #                        ACCURACY REAL,
    #                        unique (STRATEGY, DATA, SEED)
    #                    )''')
    # except sqlite3.OperationalError:
    #     pass
    #
    # data = ('Pgot', 'ENZYMES', args.seed, int(accuracy))
    # cur.execute("INSERT INTO classification VALUES (?, ?, ?, ?) "
    #             "ON CONFLICT DO UPDATE SET accuracy=excluded.accuracy;", data)
    # con.commit()
    # cur.close()
    # con.close()
