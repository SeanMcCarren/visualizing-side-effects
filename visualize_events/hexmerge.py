import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from plotly.colors import sample_colorscale
import plotly.graph_objects as go

ANGLE = - np.pi / 6
RADIUS = 1/np.sqrt(3)
ANGLES = np.array([ANGLE + k * (np.pi/3) for k in range(6)])
X_HEX = RADIUS * np.cos(ANGLES)
Y_HEX = RADIUS * np.sin(ANGLES)

class HexMerge:
    def __init__(self,
        target_hexes = 1000,
        kmax = 0,
        same_leaf_factor = 3,
        adjacency_factor = 1,
        distance_factor = 10,
        hexes_scaling = lambda x : x,
        temperature = lambda k, kmax: (1 - (k)/kmax) ** 2 / 20, #.1/np.exp(k / 100).item()
        acceptance_function = lambda cost_diff, T, rv: cost_diff < 0 or np.exp((-cost_diff+0.01)/T) >= rv,
        quantile = .0,
        moved_running_avg_window = 10000,
        MOVED_LOWERBOUND = 0.03
    ):
        self.target_hexes = target_hexes
        self.kmax = kmax
        self.same_leaf_factor = same_leaf_factor
        self.adjacency_factor = adjacency_factor
        self.distance_factor = distance_factor
        self.hexes_scaling = hexes_scaling
        self.temperature = temperature
        self.acceptance_function = acceptance_function
        self.quantile = quantile
        self.moved_running_avg_window = moved_running_avg_window
        self.MOVED_LOWERBOUND = MOVED_LOWERBOUND

    def fit(self, t):
        # ---------- PLOTTING PARAMS
        x_width = np.amax(X_HEX) - np.amin(X_HEX)
        y_width = np.amax(Y_HEX) - np.amin(Y_HEX)

        n = int(np.sqrt(self.target_hexes) // 2)
        m = int(np.sqrt(self.target_hexes))
        self.n = n
        self.m = m
        assert 2 * n * m <= self.target_hexes

        scaled = [self.hexes_scaling(P_node.pred) for P_node in t.nodes.values() if P_node.pred is not None]
        sum_scaled = sum(scaled)
        to_place = []

        nodes = list(t.nodes.values())
        N = len(t)
        indx_P_nodes, self.P_nodes = [], []
        for i, node in enumerate(nodes):
            if (node.pred is not None and node.pred != 0):
                hexes = int(np.floor(self.hexes_scaling(node.pred) / sum_scaled * (2*n*m)).item())
                if hexes <= 0:
                    continue
                indx_P_nodes.append(i)
                self.P_nodes.append(node)
                for _ in range(hexes):
                    to_place.append(len(self.P_nodes) - 1)
        P_N = len(self.P_nodes)

        # to_place = [i for i, P_node in enumerate(self.P_nodes) for _ in range(int(np.floor(scaled[i] / sum_scaled * (2*n*m)).item()))]

        assert 2 * n * m >= len(to_place)

        t.add_parent_store()
        from itertools import combinations
        t.descendant_pred()
        pred_nodes = t.root.preds
        weights = dict()
        for node in t.nodes.values():
            if node is t.root:
                continue
            significant_preds = [pred for pred in node.preds if pred in self.P_nodes]
            for a, b in combinations(significant_preds, 2):
                if a == b:
                    assert False
                if id(a) > id(b):
                    a, b = b, a
                w = weights.get((a, b), 0)
                weights[(a, b)] = w + 1 * node.depth

        total_weight = sum(weights.values())

        for k, v in weights.items():
            weights[k] = (v / total_weight) * self.adjacency_factor

        # weight all adjacencies and all transitive stuff the same    
        for P_node in self.P_nodes:
            weights[(P_node, P_node)] = self.same_leaf_factor / P_N

        node_to_idx = {node: i for i, node in enumerate(self.P_nodes)}
        new_weights = dict()
        for k, v in weights.items():
            n1, n2 = k
            n1, n2 = node_to_idx[n1], node_to_idx[n2]
            new_weights[(n1, n2)] = v

        weights = new_weights

        # ------- DESIRED LOCATION --------
        anc_store = t.get_ancestors()

        dist = np.zeros((N, N))
        for i, anc1 in enumerate(anc_store):
            for j, anc2 in enumerate(anc_store[i+1:], i+1):
        #         common = anc1.intersection(anc2)
                diff = anc1.symmetric_difference(anc2)
                #d = 1/len(common)
        #         d = 1/(sum(a.depth for a in common)+1)
                d = len(diff) / (len(anc1) + len(anc2))
                dist[i, j] = d
                dist[j, i] = d

        similarity = dist[indx_P_nodes, :][:, indx_P_nodes]

        tsne = TSNE(metric="precomputed",
                    perplexity=30,
                    early_exaggeration=10)
        embedding = tsne.fit_transform(similarity)

        self.x_min, self.y_min = cartesian(0, 0, 0)
        self.x_max, self.y_max = cartesian(1, n, m)
        # TODO might need to offset
        x_scaler = MinMaxScaler(feature_range=(self.x_min,self.x_max))
        y_scaler = MinMaxScaler(feature_range=(self.y_min,self.y_max))
        embedding[:, 0] = x_scaler.fit_transform(embedding[:, 0].reshape(-1, 1)).reshape(-1)
        embedding[:, 1] = y_scaler.fit_transform(embedding[:, 1].reshape(-1, 1)).reshape(-1)

        # to_01_scaler = MinMaxScaler(feature_range=(0,1)).fit(embedding) # distorts distances because figure does not have asp ratio of 1

        def cost(assignment):
            cost_matrix_adj = np.zeros((2, n, m, 6))
            it = np.nditer(assignment, flags=['multi_index'])
            for x in it:
                if x < 0:
                    continue
                for neighbor_idx, neighbor in enumerate(self.neighbors(*it.multi_index)): # can make more eff due to symmetry 
                    y = assignment[neighbor]
                    if y < 0:
                        continue
                    w = weights.get((x.item(), y), 0)
                    if w == 0:
                        w = weights.get((y, x.item()), 0) # only one will exist atm
                    if w != 0:
                        cost_matrix_adj[(*it.multi_index, neighbor_idx)] = -w
            
            cost_matrix_dis = np.zeros((2, n, m))
            it = np.nditer(assignment, flags=['multi_index'])
            for x in it:
                if x < 0:
                    continue
                tile_x, tile_y = cartesian(*it.multi_index)
                xx, xy = embedding[x].tolist()
                d = ((xx - tile_x)/(2*n-1)) ** 2 + ((xy - tile_y)/(m-1)) ** 2
                cost_matrix_dis[it.multi_index] = d
            cost_matrix_dis *= self.distance_factor

            cost_matrix_adj_sum = np.sum(cost_matrix_adj, axis=3)
            cost_agg = np.sum(cost_matrix_adj_sum) + np.sum(cost_matrix_dis)
            return cost_agg, cost_matrix_adj, cost_matrix_adj_sum, cost_matrix_dis

        # def random_neighbor(s):
        #     # SWITCHEROO
        # #     x = np.random.choice(2), np.random.choice(n), np.random.choice(m)
        # #     y = neighbors(*x)[np.random.choice(3)]
        # #     s_new = s.copy()
        # #     s_new[x], s_new[y] = s_new[y], s_new[x]
        #     # RANDOM ASSIGNMENT
        #     s_new = s.copy()
        #     x = np.random.choice(2), np.random.choice(n), np.random.choice(m)
        #     s_new[x] = np.random.choice(P_N+1) - 1
        #     return s_new
            
        # s = np.random.choice(P_N, replace=True, size=(2, n, m))

        def cost_x(s, x):
            assignment = s[x]
            if assignment < 0:
                return [0] * 6, 0
            new_cost_x_neighbors = []
            for neighbor in self.neighbors(*x):
                neighbor_assignment = s[neighbor]
                if neighbor_assignment < 0:
                    w = 0
                else:
                    w = weights.get((assignment, neighbor_assignment), 0)
                    if w == 0:
                        w = weights.get((neighbor_assignment, assignment), 0)
                new_cost_x_neighbors.append(-w)

            xx, xy = cartesian(*x)
            yx, yy = embedding[assignment].tolist()
            d = ((xx - yx)/(2*n-1)) ** 2 + ((xy - yy)/(m-1)) ** 2
            d *= self.distance_factor
            return new_cost_x_neighbors, d

        def set_cost_x(s, cost_matrix, cost_matrix_sum, x, cost_neighbors_x):
            # own
            cost_matrix[x] = cost_neighbors_x
            cost_matrix_sum[x] = np.sum(cost_matrix[x])
            # others'
            for neighbor_idx, neighbor in enumerate(self.neighbors(*x)):
                self_idx = (neighbor_idx + 3) % 6
                cost_matrix[(*neighbor, self_idx)] = cost_neighbors_x[neighbor_idx]
                cost_matrix_sum[neighbor] = np.sum(cost_matrix[neighbor])

        def random_grid_point():
            return (np.random.choice(2), np.random.choice(n), np.random.choice(m))

        # RANDOM LAYOUT
        # for _ in range(2 * n * m - len(to_place)):
        #     to_place.append(-1)
        # s = np.array(to_place)
        # np.random.shuffle(s)
        # s = s.reshape(2, n, m)

        # TODO better initialization - Greedy?
        np.random.shuffle(to_place)
        s = np.ones((2, n, m), dtype=int) * -1
        # maintain a heap of: option -> max. cost reduction

        def marginal_cost(s, x, assign):
            assert s[x] < 0
            s[x] = assign
            new_cost_x_neighbors, distance_cost = cost_x(s, x)
            marginal = 2 * sum(new_cost_x_neighbors) + distance_cost
            s[x] = -1
            return marginal

        to_place_set = set(to_place)
        options = np.zeros((2, n, m, P_N))
        for a in (0,1):
            for r in range(n):
                for c in range(m):
                    for assign in to_place_set:
                        options[a, r, c, assign] = (
                            marginal_cost(s, (a, r, c), assign)
                        )

        to_place_still = {assign: to_place.count(assign) for assign in to_place_set}
        to_place_still_total = len(to_place)
        while to_place_still_total > 0:
            a, r, c, assign = np.unravel_index(np.argmin(options, axis=None), options.shape)
            s[(a, r, c)] = assign
            options[a, r, c, :] = np.infty
            to_place_still[assign] -= 1
            to_place_still_total -= 1
            if to_place_still_total == 0:
                break
            if to_place_still[assign] == 0:
                options[:, :, :, assign] = np.infty
                to_place_set.remove(assign)
            for neighbor in self.neighbors(a, r, c):
                if s[neighbor] < 0:
                    for assign in to_place_set:
                        if options[a, r, c, assign] != np.infty:
                            options[a, r, c, assign] = (
                                marginal_cost(s, (a, r, c), assign)
                            )  

        cost_s, cost_matrix_s, cost_matrix_s_sum, cost_matrix_s_distances = cost(s)
        moved = 0
        moved_running_avg = 0.
        movable_points = None
        for k in range(self.kmax):
            if k%1000==0:
                print(f"Cost: {np.sum(cost_matrix_s_sum) + np.sum(cost_matrix_s_distances)}. Running average % moved: {moved_running_avg}")
            T = self.temperature(k, self.kmax)
            # random neighbor testing
            if k % 10 == 0:
                moveable_points = np.arange(2*n*m)[cost_matrix_s_sum.reshape(-1) >= np.quantile(cost_matrix_s_sum, self.quantile)]
            x, y = np.random.choice(moveable_points, replace=False, size=2).tolist()
            x = (x//(n*m), (x//m)%n, x%m)
            y = (y//(n*m), (y//m)%n, y%m)
            if x == y: continue
            #
            old_assignment_x = s[x]
            old_assignment_y = s[y]
            old_cost_x = np.sum(cost_matrix_s[x]) + cost_matrix_s_distances[x]
            old_cost_y = np.sum(cost_matrix_s[y]) + cost_matrix_s_distances[y]
            
        #     new_assignment = np.random.choice(P_N+1) - 1 # random choice of neighbor (different color in that hex)
            s[x], s[y] = old_assignment_y, old_assignment_x
            new_cost_x_neighbors, distance_cost_x = cost_x(s, x)
            new_cost_y_neighbors, distance_cost_y = cost_x(s, y)
            
            cost_diff = 2 * (
                sum(new_cost_x_neighbors) + distance_cost_x - old_cost_x
                + sum(new_cost_y_neighbors) + distance_cost_y - old_cost_y
            ) # times TWO because of double counting adjacency TODO half_neighbors?
            acceptance = self.acceptance_function(cost_diff, T, np.random.uniform())
            if acceptance:
                moved += 1
                just_moved = True
                set_cost_x(s, cost_matrix_s, cost_matrix_s_sum, x, new_cost_x_neighbors)
                set_cost_x(s, cost_matrix_s, cost_matrix_s_sum, y, new_cost_y_neighbors)
                cost_matrix_s_distances[x] = distance_cost_x
                cost_matrix_s_distances[y] = distance_cost_y
            else:
                just_moved = False
                s[x] = old_assignment_x
                s[y] = old_assignment_y
        #     print(x, y)
            moved_running_avg = (
                (min(self.moved_running_avg_window-1, k) * moved_running_avg + just_moved)
                /min(self.moved_running_avg_window, k+1)
            )
            if moved_running_avg <= self.MOVED_LOWERBOUND and k > self.moved_running_avg_window:
                print(f"Early stop at k={k}")
                break
        #     calc = cost(s)
        #     assert (cost_matrix_s == calc[1]).all()
        #     assert (cost_matrix_s_sum == calc[2]).all()
        #     assert (cost_matrix_s_distances == calc[3]).all()
        print(f"Moved {moved} out of {k} iters")

        # PSEUDOCODE sim annealing
        # Let s = s0
        # For k = 0 through self.kmax (exclusive):

        #     T ← temperature( 1 - (k+1)/self.kmax )
        #     Pick a random neighbour, snew ← neighbour(s)
        #     If P(E(s), E(snew), T) ≥ random(0, 1):
        #         s ← snew

        # Output: the final state 

        assignment = s

        # locations = [center(i, j) for i in range(10) for j in range(10)]

        self.P_nodes_to_grid = {P_node: [] for P_node in self.P_nodes}

        it = np.nditer(assignment, flags=['multi_index'])
        for x in it:
            if x >= 0:
                self.P_nodes_to_grid[self.P_nodes[x]].append(it.multi_index)

        t.attr_label()
        self.depths = t.get_depths()    

        colors_required = P_N + max(len(V_d) for V_d in self.depths)

        self.colors = sample_colorscale(colorscale='Turbo', samplepoints=colors_required)
        np.random.shuffle(self.colors)

        # max_pred = max([P_node.pred for P_node in self.P_nodes])

    def neighbors(self, a, r, c):
        # interesting fact: if a and b are neighbors, and a is the x-th neighbor of b, then b is the (x+3%6)-th neighbor of a
        return (
            (a, r, (c+1) % self.m),
            (1-a, (r-(1-a)) % self.n, (c+a) % self.m),
            (1-a, (r-(1-a)) % self.n, (c-(1-a)) % self.m),
            (a, r, (c-1) % self.m),
            (1-a, (r+a) % self.n, (c-(1-a)) % self.m),
            (1-a, (r+a) % self.n, (c+a) % self.m)
        )

    def make_fig(self):
                
        fig = go.Figure()

        # Update axes properties
        fig.update_xaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[self.x_min - RADIUS, self.x_max + RADIUS]
        )

        fig.update_yaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            scaleanchor = "x",
            scaleratio = 1,
            range=[self.y_min - RADIUS, self.y_max + RADIUS]
        )

        # ------- CREATE CONSTRAINTS --------
        P_node_to_color = {P_node: color for P_node, color in zip(self.P_nodes, self.colors)} # shouldn't matter that colors is longer
        d_P_node_occupancy = [{P_node: [] for P_node in self.P_nodes} for _ in self.depths]
        v_to_color = dict()
        for d in range(len(self.depths)):
            V_d = self.depths[d]
            # matrix of shape (V_d)
            # largest colors maintain.
            area_coverage = np.zeros(len(V_d))
            for i, v in enumerate(V_d):
                Rv = [pred for pred in v.preds if pred in self.P_nodes]
                coverage = []
                for P_node in Rv:
                    d_P_node_occupancy[d][P_node].append(v)
                    coverage += self.P_nodes_to_grid[P_node]
                area_coverage[i] = len(set(coverage))
            indxs = np.flip(np.argsort(area_coverage))
            
            remaining_colors = set(self.colors)
            for indx in indxs:
                v = V_d[indx]
                Rv = [pred for pred in v.preds if pred in self.P_nodes]
                max_col = -1
                max_size = -1
                for P_node in Rv:
                    size = len(self.P_nodes_to_grid[P_node])
                    col = P_node_to_color[P_node]
                    if size > max_size and col in remaining_colors:
                        max_size = size
                        max_col = col
                if max_col == -1:
                    max_col = remaining_colors.pop()
                else:
                    remaining_colors.remove(max_col)
                v_color = max_col
                v_to_color[v] = v_color


        v_to_trace = dict()
        for d, V_d in enumerate(self.depths):
            for v in V_d:
                Rv = [pred for pred in v.preds if pred in self.P_nodes]
                x_coords = []
                y_coords = []
                for P_node in Rv:
        #     color = colors[node_colors[0]]
                    occupancy = d_P_node_occupancy[d][P_node]
                    order = occupancy.index(v)
                    radius = 1 - (order / len(occupancy))
                    for j, grid_coord in enumerate(self.P_nodes_to_grid[P_node]):
                        if j > 0:
                            x_coords.append(None)
                            y_coords.append(None)
                        location = cartesian(*grid_coord)
                        x_coord, y_coord = path(*location, radius_modifier = radius)
                        x_coords.extend(x_coord)
                        y_coords.extend(y_coord)
                v_to_trace[v] = len(fig.data)
                fig.add_trace(go.Scatter(x=x_coords, y=y_coords, fill="toself", mode="text", fillcolor=v_to_color[v], name=v.label, visible=False)) # mode="lines"

        nodes_left = sum(len(V_d) for V_d in self.depths)
        nodes_seen = 0  
        steps = []
        for d, V_d in enumerate(self.depths):
            nodes_this_layer = len(V_d)
            nodes_left -= nodes_this_layer
            visible = [False] * nodes_seen + [True] * nodes_this_layer + [False] * nodes_left
            nodes_seen += nodes_this_layer
            for P_node in self.P_nodes:
                if P_node.depth < d:
                    visible[v_to_trace[P_node]] = True

            step = dict(
                method="update",
                args=[{"visible": visible},
                    {"title": "Depth: " + str(i)}],  # layout attribute
            )
            steps.append(step)
                
        sliders = [dict(
            active=len(self.depths) - 1,
            currentvalue={"prefix": "Depth: "},
            pad={"t": 50},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=-0.99
            )
        )

        config = {'responsive': False}

        # go.FigureWidget(data=fig)

        # fig.show(config=config)
        return fig
    
    
    def simplify_grid_coords(self, grid_coords):
    #     edges = [
    #         grid_coord for grid_coord in grid_coords if not all(
    #             neighbor in grid_coords for neighbor in neighbors(*grid_coord)
    #         )
    #     ]
    #     while len(edges) != 0
        disc = 1000000
        segments = dict()
        for grid_coord in grid_coords:
            cart = cartesian(*grid_coord)
            x_coords, y_coords = path(*cart)
            for i, neigh in enumerate(self.neighbors(*grid_coord)):
                if neigh not in grid_coords:
                    x1, x2 = x_coords[i:i+2]
                    y1, y2 = y_coords[i:i+2]
                    x1 = int(round(x1 * disc))
                    x2 = int(round(x2 * disc))
                    y1 = int(round(y1 * disc))
                    y2 = int(round(y2 * disc))
                    adj = segments.get((x1, y1), [])
                    adj.append((x2, y2))
                    segments[(x1, y1)] = adj
        
        prev_point = list(segments.keys())[0]
        cycles = []
        current_cycle = [prev_point]
        while True:
            next_points = segments.get(prev_point, None)
            if next_points is None:
                cycles.append(current_cycle)
                if len(segments) == 0:
                    break
                prev_point = list(segments.keys())[0]
                current_cycle = []
            elif len(next_points) == 1:
                del segments[prev_point]
                prev_point = next_points[0]
            else:
                prev_point = next_points.pop() # it matters which we pop!?
            
            current_cycle.append(prev_point)
        x_coord, y_coord = [], []
        for i, cycle in enumerate(cycles):
            if i > 0:
                x_coord.append(None)
                y_coord.append(None)
            for x_co, y_co in cycle:
                x_coord.append(x_co / disc)
                y_coord.append(y_co / disc)
        return x_coord, y_coord


def center(row, col):
    cx = (col * RADIUS * 1.5)
    cy = np.sqrt(3) * row
    if col % 2 == 1:
        cy = cy + np.sqrt(3) * 0.5
    return cx, cy

def cartesian(a, r, c):
    return (
        a/2 + c,
        np.sqrt(3) * (a/2 + r)
    )

def svg_path(xc, yc, radius_modifier = 1.):
    return "M" + "L".join([str(px) + "," + str(py) for px,py in zip(xc + X_HEX * radius_modifier, yc + Y_HEX * radius_modifier)]) + "Z"

def path(xc, yc, radius_modifier = 1.):
    x_coords = (xc + X_HEX * radius_modifier).tolist()
    x_coords.append(x_coords[0])
    y_coords = (yc + Y_HEX * radius_modifier).tolist()
    y_coords.append(y_coords[0])
    return x_coords, y_coords

