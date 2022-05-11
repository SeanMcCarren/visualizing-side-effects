from yaml import load
from visualize_events.snomed import *
import pytest

def test_height():
    T = load_dag()
    assert T.height > 10 # should be 17 actually

def test_root():
    T = load_dag()
    assert T.root.name == 404684003
    assert T.root.children[0].name != 404684003

def test_single_root():
    T = load_dag()
    assert len(T.nodes) == len([t for t in T.traverse(raise_on_visited=False)])

@pytest.mark.parametrize("mode", ['least_popular_parent', 'most_popular_parent', 'random'])
def test_treeify(mode):
    T = load_dag()
    assert not T.is_tree()
    T.make_tree(mode=mode)
    [t for t in T.traverse(yield_first_visit=False, yield_visited=True, raise_on_visited=True)]
    assert T.is_tree()

def test_newick():
    T =load_dag()
    T.make_tree(mode='random')
    newick = T.get_newick()
    assert len(newick) > 0 and isinstance(newick, str)

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
