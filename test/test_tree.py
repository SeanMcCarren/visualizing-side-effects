from visualize_events.snomed import *

def test_height():
    T = load_tree()
    assert T.height == 16
