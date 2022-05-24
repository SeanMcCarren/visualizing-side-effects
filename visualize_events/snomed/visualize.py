import matplotlib.pyplot as plt
import networkx as nx
from .dag import Node, DAG

def draw_subgraph_to_leaf(x: Node, T: DAG, ax = None):
    anc = [a for a in x.ancestors()]

    edges = [
        (node.name, child.name) for node in anc for child in node.children if child in anc
    ]

    G = nx.DiGraph()
    G.add_edges_from(edges, node_size=1)

    pos = {}
    row = {}
    labels = {}
    for n in anc:
        labels[n.name] = str(n)
        nr_row = row.get(n.depth, [])
        pos[n.name] = (len(nr_row), - float(n.depth))
        row[n.depth] = nr_row
        row[n.depth].append(n)

    for n in anc:
        if len(n.parents) > 0:
            xx, yy = pos[n.name]
            pos[n.name] = (((xx-0) * -1 + (len(row[n.depth])-1-xx))/len(row[n.depth]), yy + (xx % 3 - 1)/3)

    pos[T.root.name] = (pos[T.root.name][0], pos[T.root.name][1]-0.3)
    pos[x.name] = (pos[x.name][0], pos[x.name][1] + 0.5)

    nx.draw(G, pos, with_labels=False, node_size=5, edge_color='r', alpha=0.5, ax=ax)

    for k in pos:
        xx, yy = pos[k]
        pos[k] = (xx, yy-0.05)

    nx.draw_networkx_labels(G,pos,labels,font_size=12,font_color='black', ax=ax)
    return ax


def draw_dag(T: DAG, ax = None):
    edges = [
        (node.name, child.name) for node in T.nodes.values() for child in node.children
    ]
    G = nx.DiGraph()
    G.add_edges_from(edges, node_size=1)

    rows = T.get_depths()
    pos = {}
    for d, nodes in enumerate(rows):
        if d == 0:
            for node in nodes:
                pos[node.name] = (1/2, -d)
            continue
        nodes_with_x = []
        for i, node in enumerate(nodes):
            parent_x = [pos[par.name][0] for par in node.parents for _ in range(node.d - par.d)]
            nodes_with_x.append((node, sum(parent_x) / len(parent_x)))
        nodes_with_x = sorted(nodes_with_x, key=lambda x: x[1])
        for i, node_w_x in enumerate(nodes_with_x):
            node = node_w_x[0]
            pos[node.name] = ((1+i) / (1+len(nodes)), -d)

    nx.draw(G, pos, with_labels=False, node_size=5, edge_color='r', alpha=0.5, ax=ax, arrows=False)
    plt.show()