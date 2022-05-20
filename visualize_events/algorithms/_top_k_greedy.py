import numpy as np

class heap:
    """
    Entries should be "(item, value)"
    """
    def __init__(self, values):
        self.values = sorted(values, key=lambda tuple: -tuple[1])

    def add(self, value):
        self.values.append(value)
        self.values = sorted(self.values, key=lambda tuple: -tuple[1])
    
    def pop(self):
        return self.values.pop(0)
    
    def __str__(self):
        return str(self.values)


def greedy_plus(G, k):
    V = list(G.nodes.values())
    S = []
    dist_from_S = {n: np.inf for n in V}
    H = heap([(v, compute_marginal_gain(v, dist_from_S)) for v in V])
    while len(S) < k:
        v, d_max = H.pop()
        marginal = compute_marginal_gain(v, dist_from_S)
        if d_max > marginal:
            H.add((v, marginal))
            continue
        S.append(v)

        # update dist(S,v)
        dist_from_S[v] = 0
        Q = [v]
        visited = set([v])
        while len(Q) != 0:
            node = Q.pop()
            for child in node.children:
                if dist_from_S[node] + 1 < dist_from_S[child]:
                    dist_from_S[child] = dist_from_S[node] + 1
                    if child not in visited:
                        Q.append(child)
                        visited.add(child)
    return S

def compute_marginal_gain(x, dist_from_S):
    # PSEUDOCODE WAS BAD :)
    if dist_from_S[x] == 0:
        return 0
    gain = 0
    # pseudocode forgot this

    dist_from_x = {x: 0}
    Q = [x]
    visited = set([x])
    while len(Q) != 0:
        v = Q.pop()
        if v.pred is not None and v.pred != 0.:
            if dist_from_S[v] > dist_from_x[v]: # probably redundant because of pruning
                old_gain = v.pred / (1+dist_from_S[v])
                new_gain = v.pred / (1+dist_from_x[v]) # in the pseudocode it is dist<v,u>, but this must be a mistake!
                gain += (new_gain - old_gain)

        for u in v.children:
            if u not in visited:
                dist_from_x[u] = dist_from_x[v] + 1 # BFS first visit so we needn't do min(dist_from_x[u], dist_from_x[v] + 1) I THINK # TODO check this
                if dist_from_S[u] > dist_from_x[u]:
                    # pruning, needn't consider if it is already closer to other
                    Q.append(u)
                    visited.add(u)
    return gain