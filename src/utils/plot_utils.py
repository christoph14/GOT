import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def plot_graphs_ch(x_ch, y_ch, P_true, P_est=None, pos=None):
    if P_est is None:
        P_est = np.eye(*P_true.shape)

    N_nodes = x_ch.shape[0]
    A1 = -x_ch.copy()
    np.fill_diagonal(A1, 0)
    G1 = nx.from_numpy_array(A1)

    A2 = -y_ch.copy()
    np.fill_diagonal(A2, 0)
    G2 = nx.from_numpy_array(P_true.T @ A2 @ P_true)

    n1 = np.arange(N_nodes)
    n2 = (P_est.T @ P_true @ n1).astype(int)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].axis('off')
    ax[1].axis('off')

    pos = nx.kamada_kawai_layout(G1)

    show_network(G1, pos=pos, ax=ax[0])
    show_network(G2, pos=pos, labels=n2, y=np.ones(N_nodes), ax=ax[1])

    # fig.savefig('graph2.png', dpi=fig.dpi, pad_inches=0, bbox_inches='tight')

    plt.show()


def show_network(G, y=None, labels=None, pos=None, ax=None, figsize=(5, 5)):
    if ax is None:
        plt.figure(figsize=figsize)  # image is 8 x 8 inches
        plt.axis('off')
        ax = plt.gca()

    if pos is None:
        pos = nx.kamada_kawai_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.8, cmap=plt.cm.RdYlGn, node_color=y, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)

    if labels is None:
        nx.draw_networkx_labels(G, pos, font_color='w', font_weight='bold', font_size=15, ax=ax)
    else:
        labeldict = {}
        for i, v in enumerate(G.nodes):
            labeldict[v] = labels[i]
        nx.draw_networkx_labels(G, pos, font_color='w', font_weight='bold', font_size=15, labels=labeldict, ax=ax)
