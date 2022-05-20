from visualize_events.algorithms._top_k_greedy import greedy_plus, compute_marginal_gain
import pandas as pd
from visualize_events.snomed import DAG

def example_dag():
    nodes = pd.DataFrame(
        {
            "name": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            "label": ["disease", 1, "cancer", "flu", 4, 5, 6, 7, 8, 9, 10, 11, 12, "covid"],
        }
    ).set_index("name")
    edges = pd.DataFrame(
        {
            "parent": [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4],
            "child": [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        }
    )
    predictions = pd.Series(
        [10, 0, 0, 10, 5, 20, 20, 25, 25, 30, 25, 20, 0, 200],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    )
    G = DAG(nodes, edges)
    G.set_predictions(predictions, discard_others=False)
    G.attr_label()
    return G

def test_example_paper():
    G = example_dag()
    S = greedy_plus(G, 4)
    assert S is not None
    # assert something

def test_compute_marginal():
    G = example_dag()
    dist_from_S = {node: dist for node, dist in zip(G.nodes.values(),[0,1,1,1,2,2,2,2,2,2,2,2,3,3])}
    marginal = compute_marginal_gain(list(G.nodes.values())[3], dist_from_S)
    assert marginal == 12.5
