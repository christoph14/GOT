import argparse
import os

import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.datasets import TUDataset

from utils.dataset import tud_to_networkx

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluates graph alignment algorithms.')
    parser.add_argument('dataset', type=str, help='the benchmark data set')
    parser.add_argument('--seed', type=int, default=403371, help='the used random seed')
    parser.add_argument('--path', type=str, default='../distances/', help='the path to store the output files')
    parser.add_argument('--max_graphs', type=int, default=None, help='the maximum number of graphs.')
    parser.add_argument(
        '--same_size',
        action='store_const', const=True, default=False,
        help='allow soft assignment instead of a permutation matrix'
    )
    args = parser.parse_args()

    # Load graph data set
    print(f"Dataset: {args.dataset}")
    dataset = TUDataset('../datasets', name=args.dataset)
    print(dataset)
    graphs = tud_to_networkx(args.dataset)

    if args.dataset in ['FRANKENSTEIN', 'DHFR']:
        values, counts = np.unique([G.number_of_nodes() for G in graphs], return_counts=True)
        n_nodes = values[np.argmax(counts)]
        graphs = [G for G in graphs if G.number_of_nodes() == n_nodes]
        print(f"Use only graphs of size {n_nodes}")

    # Create numpy array
    X = np.empty(len(graphs), dtype=object)
    X[:] = graphs

    # Sample graphs
    n_graphs = {
        'BZR': 405,
        'MUTAG': 188,
        'PTC_MR': 344,
        'KKI': 83,
        'ENZYMES': 300,
        'PROTEINS': 200,
        'AIDS': 500,
        'FRANKENSTEIN': 500,
        'DHFR': 500,
    }
    if n_graphs[args.dataset] < len(graphs):
        rng = np.random.default_rng(args.seed)
        X = rng.choice(X, n_graphs[args.dataset], replace=False)

    y = np.array([G.graph['classes'] for G in X])
    print(f"Seed: {args.seed}")
    print(f'{len(X)} graphs')
    _, labels = np.unique(y.flatten(), return_counts=True)
    print('Class occurrences:', labels / np.sum(labels) * 100)
    print('Average number of nodes:', np.round(np.mean([g.number_of_nodes() for g in X]), 2))
    print('Average number of edges:', np.round(np.mean([g.number_of_edges() for g in X]), 2))
    print('Possible number of nodes:', np.unique([g.number_of_nodes() for g in X]))
    values, counts = np.unique([g.number_of_nodes() for g in X], return_counts=True)
    print(max(values))
    os.makedirs('../plots/dataset_node_distributions', exist_ok=True)
    scale = 0.3
    fig, ax = plt.subplots(figsize=(6*scale, 4*scale))
    ax.bar(values, counts)
    # plt.tight_layout(pad=0)
    plt.savefig(f'../plots/dataset_node_distributions/{args.dataset}.pdf', bbox_inches='tight')
