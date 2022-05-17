from visualize_events.snomed import *
from visualize_events.data import *
import pytest

# def test_height(): # very expensive haha
#     T = load_dag()
#     assert T.height > 10 # should be 17 actually

def test_root():
    T = load_dag()
    assert T.root.name == 138875005
    assert T.root.children[0].name != 138875005

def test_single_root():
    T = load_dag()
    assert len(T.nodes) == len([t for t in T.traverse(raise_on_visited=False)])

# @pytest.mark.parametrize("mode", ['least_popular_parent', 'most_popular_parent', 'random'])
# def test_treeify(mode):
#     T = load_dag()
#     assert not T.is_tree()
#     T.make_tree(mode=mode)
#     [t for t in T.traverse(yield_first_visit=False, yield_visited=True, raise_on_visited=True)]
#     assert T.is_tree()

def test_newick():
    T = load_dag()
    preds = get_predictions(861)
    T2 = T.set_predictions(preds, discard_others=True)
    newick = T2.get_newick(depth=3)
    assert len(newick) > 0 and isinstance(newick, str)
    from ete3 import Tree
    t = Tree(newick, format=8, quoted_node_names=True)
    some = False
    for n in t.traverse():
        if hasattr(n, 'agg') and hasattr(n, 'f'):
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

def small_dag():
    import pandas as pd
    

def test_compact():
    T = load_dag()
    preds = get_predictions(861)
    T = T.set_predictions(preds)
    T.add_parent_store()
    root = T.compact_preds()
    print(root)

test_compact()