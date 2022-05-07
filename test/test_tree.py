from visualize_events.snomed import *

def test_height():
    T = load_tree()
    assert T.height == 16

def test_root():
    T = load_tree()
    assert T.root.name == 138875005
    assert T.root.children[0].name != 138875005

def test_weights():
    T = load_tree()
    T.remove_other_roots(verbose=True)
    T.make_tree(mode='most_popular_parent')
    assert T.root.w > 0

def test_treeify():
    T = load_tree()
    T.attr_label()
    T.remove_other_roots(verbose=True)
    assert not T.is_tree()
    T.make_tree(mode='most_popular_parent')
    assert T.is_tree()
    print(T.get_newick())
