from multiprocessing.sharedctypes import Value
import pickle
from typing import Optional, List
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import copy

from visualize_events.data import load_data

logger = logging.getLogger()

class Node:
    def __init__(self, name: int, label: Optional[str] = None):
        assert isinstance(name, (int, np.int64))
        self.name = int(name)
        self.children = []
        self.important = False
        self.w = None  # Can hold weight ?
        self.pred:float = None  # Can hold predictions
        self.preds:NodeSet(Node) = None # Can hold descendant-predictions
        self.pred_agg:float = None  # Can hold aggregated prediction score
        self.d = None  # Can hold depth value
        self.label = label  # Holds label of concept
        self.leafs = None  # Can hold nr of leafs
        self.visited = False

    @property
    def height(self):
        if len(self.children) == 0:
            return 0
        else:
            return max((child.height for child in self.children)) + 1

    @property
    def depth(self):
        if self.d is None:
            if len(self.parents) == 0: # must have set parents!
                self.d = 0
            else:
                self.d = max((parent.depth for parent in self.parents)) + 1
        return self.d

    @property
    def size(self):
        if len(self.children) == 0:
            return 1
        else:
            return sum((child.size for child in self.children)) + 1

    @property
    def has_important(self):
        if self.important:
            return True
        elif len(self.children) == 0:
            return self.important
        else:
            return any((child.has_important for child in self.children))

    def make_important(self):
        if self.important:
            return
        else:
            self.important = True
            for c in self.children: c.make_important()

    def aggregate_pred(self, agg_func):
        # avoid repetitive work in non-tree DAG
        if self.pred_agg is not None:
            return self.pred_agg

        child_preds = [c.aggregate_pred(agg_func) for c in self.children]
        if self.pred is not None:
            child_preds.append(self.pred)
        if len(child_preds) == 0:
            raise ValueError("No preds in subtree")
        arr = np.array(child_preds)
        self.pred_agg = agg_func(arr, axis=0)
        return self.pred_agg

    def descendant_pred(self):
        if self.preds is not None:
            return self.preds

        self.preds = NodeSet([pred for child in self.children for pred in child.descendant_pred()])
        if self.pred is not None:
            self.preds.add(self)
        return self.preds

    def pop_child(self, node):
        if node in self.children:
            self.children.pop(self.children.index(node))

    def add_child(self, node):
        if node not in self.children:
            self.children.append(node)

    def pop_parent(self, node):
        if node in self.parents:
            self.parents.pop(self.parents.index(node))

    def add_parent(self, node):
        if node not in self.parents:
            self.parents.append(node)

    def __repr__(self):
        if hasattr(self, 'parents'):
            return f"P:{len(self.parents)}, C:{len(self.children)} NODE: {str(self)}"
        else:
            return f"C:{len(self.children)} NODE: {str(self)}"

    def __str__(self):
        if self.label is not None:
            return str(self.label)
        else:
            return str(self.name)

    def get_newick(self, depth=None):
        string = ""
        if len(self.children) != 0 and (depth is None or depth > 0):
            string = (
                "("
                + ",".join(
                    [
                        c.get_newick(depth if depth is None else depth - 1)
                        for c in self.children
                    ]
                )
                + ")"
            )
        name = str(self)
        if '"' in name:
            raise ValueError("Will error!")
        string += '"' + str(self) + '"'
        args = []
        if self.pred is not None:
            args.append(f"f={self.pred}")
        if self.pred_agg is not None:
            args.append(f"agg={self.pred_agg}")
        if len(args) != 0:
            args = ":".join(args)
            string += f"[&&NHX:{args}]"
        return string

    def ancestors(self):
        to_visit = [self]
        visited_or_going_to = set([self])
        while to_visit:
            n = to_visit.pop(0)
            yield n
            for p in n.parents:
                if p not in visited_or_going_to:
                    to_visit.append(p)
                    visited_or_going_to.add(p)

    def descendants(self):
        """
        BFS down
        """
        to_visit = [self]
        visited_or_going_to = set([self])
        while to_visit:
            n = to_visit.pop(0)
            yield n
            for c in n.children:
                if c not in visited_or_going_to:
                    to_visit.append(c)
                    visited_or_going_to.add(c)

    def iter_leafs(self):
        for des in self.descendants():
            if len(des.children) == 0:
                yield des

    def reset_visited_children(self):
        self.visited = False
        for c in self.children: c.reset_visited_children()

    def reset_visited_parents(self):
        self.visited = False
        for p in self.parents: p.reset_visited_parents()

    def reset_depth(self):
        for node in self.nodes.values():
            node.d = None

    def _attr_ancs(self, ancs=None):
        """
        We borrow self.leafs
        """
        if ancs == None:
            ancs = NodeSet([self])
        else:
            ancs = ancs.copy()
            ancs.add(self)

        propagate = True
        if self.leafs is None:
            self.leafs = ancs
        else:
            N = len(self.leafs)
            self.leafs.update(ancs)
            if N == len(self.leafs):
                propagate = False

        if propagate:
            for c in self.children:
                c._attr_ancs(ancs)

    def dist_up(self, other):
        # use BFS!
        # dist_up is likely better than dist_down, because a node will have less ancestors than descendants!
        to_visit = [(self, 0)]
        visited = set()
        while len(to_visit) != 0:
            n, d = to_visit.pop(0)
            if n is other:
                return d
            else:
                for p in n.parents:
                    if p not in visited:
                        to_visit.append((p, d+1))
                        visited.add(p)




primes = [i for i in range(2, 1000)]
for iter in range(1, 100):
    if iter >= len(primes):
        break
    prime = primes[iter]
    primes = primes[:iter] + [p for p in primes[iter:] if p % prime == 0]

class NodeSet(set):
    def __hash__(self):
        names = [n.name for n in self]
        return sum(name * prime for name, prime in zip(sorted(names), primes))

class NodeGroup():
    def __init__(self, nodes=None):
        if nodes is None:
            nodes = NodeSet()
        self.nodes = nodes # nodes in original DAG
        self.children = [] # children groups

    @property
    def name(self):
        # get name from nodes
        # ASSUME most representative node is the deepest. HOWEVER does that work if
        # there is an intermediate prediction? I guess then it would fall
        # under a different node group?
        if len(self.nodes) == 0:
            raise ValueError("Has no name because has no nodes")
        some_node = None
        for node in self.nodes:
            some_node = node
            break
        has_more = True
        while has_more:
            has_more = False
            for c in some_node.children:
                if c in self.nodes:
                    has_more = True
                    some_node = c
        return some_node.name

    def unique_leafs(self):
        child_leafs = set()
        for c in self.children:
            child_leafs.update(c.leafs)
        return self.leafs - child_leafs

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.add(node)

    def __repr__(self):
        return repr([n.name for n in self.nodes])

class DAG:
    """
    Actually a DAG with a single source
    """

    _WARNED = False
    _PARENTS_COMPUTED = False

    def __init__(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        edge_types: pd.Series=None,
        single_source=True,
        discard_singles=True,
    ):
        self.nodes = dict()
        # Explicitely not copying the dataframes to save time
        self.nodes_df = nodes
        self.edges_df = edges
        self.edge_types = edge_types
        self.prediction_df = None
        self.adjacency_df = None
        self.root = None

        if discard_singles:
            self._discard_singles()

        # Add nodes
        assert (self.nodes_df.reset_index().columns == ["name", "label"]).all()
        for name, label in self.nodes_df.reset_index().values:
            self.nodes[name] = Node(name)

        # Add edges
        assert (self.edges_df.columns[:2] == ["parent", "child"]).all()
        grouped = self.edges_df.groupby('parent')['child'].apply(set)
        for parent, children in grouped.reset_index().values:
            parent_node = self.nodes[parent]
            parent_node.children = list(self.nodes[child] for child in children)
        # Slower, I think.
        # for parent, child, edge_type in edges.values:
        #     parent_node = self.nodes[parent]
        #     child_node = self.nodes[child]
        #     parent_node.add_child(child_node)
        #     # child_node.add_parent(parent_node)

        # Check how many source nodes there are. If only one, this becomes root
        if single_source:
            self._compute_root()

    @property
    def height(self):
        return self.root.height

    def is_tree(self):
        try:
            if self._PARENTS_COMPUTED:
                for n in self.nodes.values():
                    if len(n.parents) > 1:
                        raise Exception("visited", n)
            else:
                for _ in self.traverse(
                    yield_first_visit=False, yield_visited=True, raise_on_visited=True
                ):
                    pass
        except Exception as e:
            if e.args is not None and len(e.args) > 0 and e.args[0] == "visited":
                print(e.args[1])
                return False
            else:
                raise e
        return True

    def get_newick(self, depth=None):
        return self.root.get_newick(depth) + ";"

    def traverse(
        self, yield_first_visit=True, yield_visited=False, raise_on_visited=True
    ):
        visited = set()
        to_visit = [self.root]
        while to_visit:
            n = to_visit.pop(-1)
            if yield_first_visit:
                yield n
            for c in n.children:
                if c not in visited:
                    to_visit.append(c)
                    visited.add(c)
                else:
                    if yield_visited:
                        yield c
                    if raise_on_visited:
                        raise Exception("visited", c)

    def __attr_nr_leafs(self):
        def attr_recursive(n):
            if n.leafs is None:
                leafs = None
                if len(n.children) == 0:
                    leafs = set([n])
                else:
                    leafs = set()
                    for c in n.children:
                        leafs.update(attr_recursive(c))
                n.w = len(leafs)
                n.leafs = leafs
            return n.leafs

        attr_recursive(self.root)
        for n in self.traverse(raise_on_visited=False):
            n.leafs = None
            assert n.w != 0

    def clear_leafs(self):
        for n in self.nodes.values():
            n.leafs = None

    def get_ancestors(self):
        self.add_parent_store()
        self.root._attr_ancs()
        anc_store = []
        for n in self.nodes.values():
            anc_store.append(n.leafs)
        self.clear_leafs()
        return anc_store

    def attr_label(self):
        for name, node in self.nodes.items():
            node.label = self.nodes_df.loc[name, 'label']

    def make_tree(self, mode="most_popular_parent"):
        self.add_parent_store()
        if mode not in ["most_popular_parent", "least_popular_parent", "random"]:
            raise ValueError(f"Mode {mode} not recognized")
        if "popular_parent" in mode:
            self.__attr_nr_leafs()
        for n in self.nodes.values():
            if mode == "most_popular_parent":
                if len(n.parents) > 1:
                    popular_parent = n.parents[0]
                    for par in n.parents[1:]:
                        if par.w > popular_parent.w:
                            popular_parent.pop_child(n)
                            popular_parent = par
                        else:
                            par.pop_child(n)
                    n.parents = [popular_parent]
            elif mode == "least_popular_parent":
                if len(n.parents) > 1:
                    popular_parent = n.parents[0]
                    for par in n.parents[1:]:
                        if par.w < popular_parent.w:
                            popular_parent.pop_child(n)
                            popular_parent = par
                        else:
                            par.pop_child(n)
                    n.parents = [popular_parent]
            elif mode == "random":
                if len(n.parents) > 1:
                    keep_i = np.random.choice(len(n.parents))
                    for i, p in enumerate(n.parents):
                        if i == keep_i:
                            n.parents == [p]
                        else:
                            p.pop_child(n)

    def del_parent_store(self):
        if not self._PARENTS_COMPUTED:
            return

        for n in self.nodes.values():
            del n.parents

        self._PARENTS_COMPUTED = False

    def add_parent_store(self):
        if self._PARENTS_COMPUTED:
            return

        for n in self.nodes.values():
            n.parents = []

        for n in self.nodes.values():
            for c in n.children:
                c.add_parent(n)

        self._PARENTS_COMPUTED = True

    def _del_node(self, n, reconnect=True):
        if not self._PARENTS_COMPUTED:
            self.add_parent_store()

        # reconnect
        if reconnect:
            for c in n.children:
                for p in n.parents:
                    p.add_child(c)
                    c.add_parent(p)
        # unlink parents
        for p in n.parents:
            p.pop_child(n)
        # unlink children
        for c in n.children:
            c.pop_parent(n)

        # delete node
        del self.nodes[n.name]
        del n

    def splice(self, keep_constraint):
        for node in list(self.nodes.values()):
            if not keep_constraint(node):
                self._del_node(node)

    def keep_clinical_findings_and_body(self):
        T = self.get_edge_types([0,1])
        body = T.root.children[-8]
        cf = T.root.children[-5]

        assert all(T.nodes_df.loc[[body.name, cf.name]]['label'] == ['Body structure (body structure)', 'Clinical finding (finding)'])

        body.make_important()
        cf.make_important()
        T.root.important = True
        return T.filter(keep_constraint = lambda n: n.important)

    def keep_clinical_findings(self):
        T = self.get_edge_types([0])
        cf = T.root.children[-5]

        assert T.nodes_df.loc[cf.name]['label'] == 'Clinical finding (finding)'

        cf.make_important()
        T.root.important = True
        return T.filter(keep_constraint = lambda n: n.important)


    def filter(self, keep_constraint=None, keep_names=None):
        # N = 0
        # if traverse:
        #     for n in self.traverse(
        #         yield_first_visit=True,
        #         yield_visited=False,
        #         raise_on_visited=True,
        #     ):
        #         if not keep_constraint(n):
        #             N += 1
        #             self._del_node(n)
        #     logger.info(f"Removed {N} nodes.")
        #     self._filter_df()
        # else:
            
        # More efficient for large number of nodes.
        assert keep_constraint is None or keep_names is None
        assert keep_constraint is not None or keep_names is not None
        if keep_constraint is not None:
            keep_names = np.array([name for name, node in self.nodes.items() if keep_constraint(node)])
        nodes_df = self.nodes_df.loc[keep_names]
        edges_df = self.edges_df[np.logical_and(np.isin(self.edges_df['child'], keep_names), np.isin(self.edges_df['parent'], keep_names))]
        return DAG(nodes_df, edges_df, self.edge_types)

    def _discard_singles(self):
        self._compute_adjacency()

        singles_idx = np.logical_and(self.adjacency_df['nr_children'] == 0, self.adjacency_df['nr_parents'] == 0)
        non_singles = self.adjacency_df.index.values[~singles_idx]
        self._filter_inplace(keep_names = non_singles)

    def _compute_root(self):
        if self.root is not None:
            return

        self._compute_adjacency()
        
        sources = np.logical_and(self.adjacency_df['nr_children'] > 0, self.adjacency_df['nr_parents'] == 0)
        if np.sum(sources) == 1:
            root_name = self.adjacency_df.index.values[sources].item()
            self.root = self.nodes[root_name]
        elif np.sum(sources) == 0:
            logger.warn("No sources! was the tree a single and did you discard it?")
        else:
            raise ValueError("Provided nodes and edges have more than one source")

    def _compute_adjacency(self):
        if self.adjacency_df is not None:
            return
        self.adjacency_df = (
            pd.DataFrame(
                data={
                    "nr_children": self.edges_df.groupby("parent")["child"].count(),
                    "nr_parents": self.edges_df.groupby("child")["parent"].count(),
                },
                index=self.nodes_df.index,
            )
            .fillna(0)
            .astype(int)
        )

    def get_subgraph(self, names):
        self.add_parent_store()

        keep_names_transitive = set()
        to_visit = []
        not_found = 0
        for name in names:
            n = self.nodes.get(name, None)
            if n is not None:
                to_visit.append(n)
            else:
                not_found += 1

        if not_found > 0:
            logger.warning(f"{not_found} nodes not found")

        while to_visit:
            n = to_visit.pop()
            if n.name not in keep_names_transitive:
                keep_names_transitive.add(n.name)
                for p in n.parents:
                    if (p.name not in keep_names_transitive) and (p not in to_visit):
                        to_visit.append(p)
        
        return self.filter(keep_names=list(keep_names_transitive))

    def _filter_inplace(self, keep_names):
        self.nodes_df = self.nodes_df[np.isin(self.nodes_df.index, keep_names)]
        self.edges_df = self.edges_df[np.isin(self.edges_df["parent"], keep_names)]
        self.edges_df = self.edges_df[np.isin(self.edges_df["child"], keep_names)]
        if self.adjacency_df is not None:
            self.adjacency_df = self.adjacency_df[np.isin(self.adjacency_df.index, keep_names)]
        # TODO also filter predictions_df

    def get_edge_types(self, types):
        edges = self.edges_df[np.isin(self.edges_df['type'], types)]
        return DAG(self.nodes_df, edges)

    def __len__(self):
        return len(self.nodes)

    def set_predictions(self, prediction_df, aggregate='sum', discard_others=True):
        assert isinstance(prediction_df, pd.Series)
        # assert prediction_df.index.name == "snomed_reaction"
        if self.root.pred_agg is not None:
            logger.warn("Predictions not reset. Consider calling reset_pred")

        N_preds = len(prediction_df)
        prediction_df = prediction_df.loc[np.isin(prediction_df.index, self.nodes_df.index)]
        if len(prediction_df) < N_preds:
            logger.warning(f"{N_preds - len(prediction_df)} predictions not found")

        if discard_others:
            subgraph = self.get_subgraph(prediction_df.index.values)
            subgraph.set_predictions(prediction_df, aggregate=aggregate, discard_others=False)
            return subgraph

        self.prediction_df = prediction_df

        for index, row in prediction_df.iteritems():
            reaction = index
            n = self.nodes[reaction]
            n.pred = row # could be a tuple!

        if aggregate != False and aggregate is not None:
            self.aggregate_pred(aggregate)

    def reset_pred(self):
        for n in self.nodes.values():
            n.pred = None
            n.pred_agg = None
            n.preds = None

    def aggregate_pred(self, aggregate):
        if aggregate == 'max':
            agg_func = np.amax
        elif aggregate == 'sum':
            agg_func = np.sum
        else:
            raise ValueError(f"Can not interpret aggregation function {aggregate}")
        self.root.aggregate_pred(agg_func)

    def descendant_pred(self):
        self.root.descendant_pred()

    def get_depths(self):
        """
        returns a list of list, where the i-th list has the nodes on depth i.
        """
        self.add_parent_store()
        depths = {}
        for node in self.nodes.values():
            if node.depth not in depths:
                depths[node.depth] = []
            depths[node.depth].append(node)
        diff_depths = sorted(list(depths.keys()))
        assert max(diff_depths) == len(diff_depths) - 1 and diff_depths[0] == 0
        return [depths[k] for k in diff_depths]

    def compact_preds(self):
        # TODO move to algorithms
        # finding leafs can be faster?
        self.descendant_pred()

        preds_sets = set(n.preds for n in self.nodes.values())
        preds_to_groups = {preds: NodeGroup() for preds in preds_sets}
        node_to_groups = dict()
        for node in self.nodes.values():
            preds = node.preds
            group = preds_to_groups[preds]
            node_to_groups[node] = group
            # additionally add node to group
            group.add_node(node)

        for n in self.nodes.values():
            n_group = node_to_groups[n]
            for c in n.children:
                c_group = node_to_groups[c]
                if n_group is not c_group and c_group not in n_group.children:
                    n_group.children.append(c_group)

        # for g in preds_to_groups.values():

        names = []
        for g in preds_to_groups.values():
            names.append(g.name)

        nodes_df = self.nodes_df.loc[names]

        parents, childs = [], []
        for group in preds_to_groups.values():
            for child_group in group.children:
                parents.append(group.name)
                childs.append(child_group.name)
            # for leaf in group.unique_leafs():
            #     parents.append(group.name)
            #     childs.append(leaf.name)

        # nodes_df = pd.DataFrame({'name': names, 'label':labels}).set_index('name', drop=True)
        edges_df = pd.DataFrame({'parent': parents, 'child': childs})

        dag = DAG(nodes_df, edges_df, discard_singles=False)

        return dag

    def copy(self, init=False):
        # Using dataframe
        if init:
            if self.prediction_df is not None:
                logger.warn("Predictions are not copied. Use .copy(init=False) instead")
            return DAG(self.nodes_df, self.edges_df, self.edge_types)

        # Using deepcopy
        else:
            self.del_parent_store()
            copied = copy.deepcopy(self)
            return copied

def load_dag(force_reload=False):
    cache_location = Path(__file__).parent / 'cached_snomed'
    if not force_reload:
        try:
            with open(cache_location, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            pass

    path = Path(
        r"C:\Users\Sean McCarren\Downloads\SnomedCT_InternationalRF2_PRODUCTION_20220430T120000Z\SnomedCT_InternationalRF2_PRODUCTION_20220430T120000Z\Full\Terminology"
    )
    # TODO better load
    import os

    (
        CONCEPT,
        DESCRIPTION,
        IDENTIFIER,
        RELATIONSHIP_CONCRETE_VALUES,
        RELATIONSHIP,
        STATED_RELATIONSHIP,
        TEXT_DEFINITION,
        SREFSET,
    ) = [str(path / file) for file in sorted(os.listdir(path))]
    description = pd.read_csv(DESCRIPTION, sep="\t", index_col="id")
    relationship = pd.read_csv(RELATIONSHIP, sep="\t")

    def newest(df):
        highest_times = df.groupby("id")["effectiveTime"].transform(max)
        newest_df = df[highest_times == df["effectiveTime"]]
        assert len(newest_df.index) == len(np.unique(newest_df.index.values))
        return newest_df

    def active(df):
        active_df = df[df["active"] == 1]
        assert (active_df["active"] == 1).all()
        return active_df

    description = active(newest(description))
    relationship = active(newest(relationship))

    FSN = description[description["typeId"] == 900000000000003001][
        ["conceptId", "term"]
    ].drop_duplicates(subset=["conceptId"])
    nodes = pd.DataFrame({"name": FSN["conceptId"], "label": FSN["term"]}).set_index(
        "name"
    )
    only_is_a = False
    if only_is_a:
        relationship = relationship[relationship["typeId"] == 116680003]
    edges = (
        relationship[["destinationId", "sourceId", "typeId"]]
        .rename(
            columns={"destinationId": "parent", "sourceId": "child", "typeId": "type"}
        )
        .drop_duplicates()
    )
    ranked_relations = edges["type"].value_counts().index.values
    mapping = pd.Series(
        nodes.loc[ranked_relations, "label"].values,
        index=np.arange(len(ranked_relations)),
    )
    inverse = pd.Series(np.arange(len(ranked_relations)), index=ranked_relations)
    edges["type"] = inverse[edges["type"]].values
    # TODO check for parallel edges? there are parallel edges.
    dag = DAG(nodes, edges, mapping)
    with open(cache_location, 'wb') as f:
        pickle.dump(dag, f)
    return dag


def load_dag_old():
    clinical_finding_edges = load_data("clinical_finding_edges")
    clinical_finding_nodes = load_data("clinical_finding_nodes")
    T = DAG(clinical_finding_nodes, clinical_finding_edges)
    return T
