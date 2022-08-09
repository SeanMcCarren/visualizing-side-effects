from visualize_events.snomed import *
from visualize_events.data import *
import pytest
import pandas as pd
import numpy as np
from visualize_events.algorithms import representative, CoverageDistance, greedy_plus

def diamond_dag():
    nodes_df = pd.DataFrame({'name':[0,1,2,3,4],'label':[0,1,2,3,4]}).set_index('name')
    edges_df = pd.DataFrame({'parent':[0, 0, 0, 1, 2, 3],'child':[1, 2, 3, 4, 4, 4]})
    T = DAG(nodes_df, edges_df)
    predictions = pd.Series([1], index=[4])
    T = T.set_predictions(predictions)
    return T

def diamond_dag_tail():
    nodes_df = pd.DataFrame({'name':[0,1,2,3,4,5],'label':[0,1,2,3,4,5]}).set_index('name')
    edges_df = pd.DataFrame({'parent':[0, 0, 0, 1, 2, 3,1],'child':[1, 2, 3, 4, 4, 4,5]})
    T = DAG(nodes_df, edges_df)
    predictions = pd.Series([1, 1], index=[4, 5])
    T = T.set_predictions(predictions)
    return T

def wide_dag():
    T = load_dag()
    disease = T.nodes[64572001]
    names = [disease.name] + [c.name for c in disease.children]
    subgraph = T.filter(keep_names = names, copy_pred=False)

    predictions = pd.Series(
        [0.4, 0.4, 0.1],
        index=[733141003, 733140002, 44730006],
    )

    T = subgraph.set_predictions(predictions)
    return T

def report_dag():
    nodes_df = pd.DataFrame({'name':
        [
            0, 1, 2, 3, 4,
            5, 6, 7, 8, 9,
            10, 11, 12, 13, 14,
            15
        ] + [i for i in range(16, 116)],
        'label':[
            'Disease', 'Eczema', 'Nausea', 'Upper body', 'Lower body',
            'Head', 'Face', 'Eczema of face', 'Legs', 'Eczema of leg',
            'Dry eczema', 'Lung disease', 'Lung infection', 'Lung cancer', 'Bronchitis',
            'Bronchiolitis',
        ] + [i-16 for i in range(16, 116)]
    }).set_index('name')
    edges_df = pd.DataFrame(
        {
            'parent':[0, 0, 1, 1, 3, 5, 6, 4, 8, 1, 0, 0, 11, 11, 12, 11, 12] + [0 for i in range(16, 116)],
            'child':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 15, 15] + [i for i in range(16, 116)]
        }
    )
    T = DAG(nodes_df, edges_df)
    predictions = pd.Series([14, 10, 10, 10, 1, 16, 15], index=[2, 7, 9, 10, 13, 14, 15])
    T_pred = T.set_predictions(predictions)
    return T, T_pred

def predictions_dag(prediction=861):
    T = load_dag()
    preds = get_predictions(prediction)
    T = T.set_predictions(preds)
    return T

def test_height(): # very expensive haha
    T = diamond_dag()
    assert T.height == 2 # should be 17 actually

def test_root():
    T = load_dag()
    assert T.root.name == 138875005
    assert T.root.children[0].name != 138875005
    assert len(T.nodes) == len([t for t in T.traverse(raise_on_visited=False)])

# @pytest.mark.parametrize("mode", ['least_popular_parent', 'most_popular_parent', 'random'])
# def test_treeify(mode):
#     T = load_dag()
#     assert not T.is_tree()
#     T.make_tree(mode=mode)
#     [t for t in T.traverse(yield_first_visit=False, yield_visited=True, raise_on_visited=True)]
#     assert T.is_tree()

def test_newick():
    T = predictions_dag()
    newick = T.get_newick(depth=3)
    assert len(newick) > 0 and isinstance(newick, str)
    from ete3 import Tree
    t = Tree(newick, format=8, quoted_node_names=True)
    some = False
    for n in t.traverse():
        if hasattr(n, 'f'):
            some = True
            break
    assert some

@pytest.mark.parametrize("init", [True, False])
def test_copy(init):
    G = load_dag()
    T = G.copy(init=init)
    assert len(T) == len(G)
    T_traversal = [t for t in T.traverse(raise_on_visited=False)]
    G_traversal = [t for t in G.traverse(raise_on_visited=False)]
    assert len(T_traversal) == len(G_traversal)
    for a, b in zip(T_traversal, G_traversal):
        assert a is not b
        assert a.name == b.name

def test_summary_graph():
    T = predictions_dag(None)
    # T = diamond_dag()
    draw_nodes = representative(T, 50)
    summary = T.summary_graph(draw_nodes)
    print(summary)

def test_example_report():
    T, T_P = report_dag()
    assert len(T) == 116
    assert len(T_P) == 16
    T_P.attr_label()
    ours = ['Lung infection', 'Nausea', 'Eczema']

    cov = CoverageDistance(T, func=np.sqrt, only_leafs=False)
    draw_nodes = cov.greedy(T_P, 3)
    assert set([n.label for n in draw_nodes]) == set(ours)
    
    cov = CoverageDistance(T, func=np.sqrt, only_leafs=True)
    draw_nodes = cov.greedy(T_P, 3)
    assert set([n.label for n in draw_nodes]) == set(ours)

    theirs = ['Disease', 'Bronchitis', 'Bronchiolitis']
    draw_nodes = greedy_plus(T_P, 3)
    assert set([n.label for n in draw_nodes]) == set(theirs)
