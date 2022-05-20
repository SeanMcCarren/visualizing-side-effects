import numpy as np

def representative(G, k, score = lambda w, d: w * 1/(1.9**d)):
    anc_store = G.get_ancestors()

    representative_scores = {node: 0. for node in G.nodes.values()}  
    for node, ancs in zip(G.nodes.values(), anc_store):
        if node.pred is not None and node.pred is not 0.:
            w = node.pred # could transform this somehow! for instance np.log(w/threshold)
            
            for anc in ancs:
                representative_scores[anc] += score(w,node.dist_up(anc))

    values = np.array(list(representative_scores.values()))
    k = min(len(values), k)
    ind = np.argpartition(values, -k)[-k:]
    draw_nodes = np.array(list(representative_scores.keys()))[ind].tolist()
    return draw_nodes