from typing import Optional
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import copy

from pyrsistent import discard

from visualize_events.data import load_data

logger = logging.getLogger()

class Node:
    def __init__(self, name: int, label: Optional[str] = None):
        assert isinstance(name, int)
        self.name = name
        self.parents = []
        self.children = []
        self.important = False
        self.w = None # Can hold weight ?
        self.pred = None # Can hold predictions
        self.d = None # Can hold depth value
        self.label = label # Holds label of concept
        self.leafs = None # Can hold nr of leafs

    @property
    def height(self):
        if len(self.children) == 0:
            return 0
        else:
            return max((child.height for child in self.children)) + 1

    @property
    def depth(self):
        if len(self.parents) == 0:
            return 0
        else:
            return max((parent.depth for parent in self.parents)) + 1
    
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
    
    def del_subdag(self):
        for par in self.parents:
            par.pop_child(self)
        for c in self.children:
            c.del_subdag()
        del c # this does not maintain dag's nodes
    
    def __repr__(self):
        return f"NODE: {self.name}. P:{len(self.parents)}, C:{len(self.children)}"

    def __str__(self):
        if self.label is not None:
            return str(self.label)
        else:
            return str(self.name)

    def get_newick(self, depth=None):
        string = ""
        if len(self.children) != 0 and (depth is None or depth > 0):
            string = "(" + ",".join([c.get_newick(depth if depth is None else depth-1) for c in self.children]) + ")"
        name = str(self)
        if '"' in name:
            raise ValueError("Will error!")
        string += '"' + str(self) + '"'
        return string

    def ancestors(self):
        yield self
        visited = set([self])
        for p in self.parents:
            for anc in p.ancestors():
                if anc not in visited:
                    yield anc
                    visited.add(anc)
    
    def descendants(self):
        yield self
        visited = set([self])
        for c in self.children:
            for des in c.ancestors():
                if des not in visited:
                    yield des
                    visited.add(des)

        
class DAG:
    """
    Actually a DAG with a single source
    """
    _WARNED = False

    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame):
        self.nodes = dict()
        self.nodes_df = nodes.copy()
        self.edges_df = edges.copy()
        self.prediction_df = None

        # Add nodes
        for name, label in nodes.reset_index().values:
            self.nodes[name] = Node(name, label=label) # Need not add label for production system

        # Add edges
        for parent, child in edges.values:
            parent_node = self.nodes[parent]
            child_node = self.nodes[child]
            parent_node.add_child(child_node)
            child_node.add_parent(parent_node)

        # Check how many source nodes there are. If only one, this becomes root
        sources = [node for node in self.nodes.values() if len(node.parents) == 0]
        if len(sources) == 1:
            self.root = sources[0]
        else:
            raise ValueError("Provides nodes and edges have more than one source")
    
    @property
    def height(self):
        return self.root.height

    def is_tree(self):
        try:
            for _ in self.traverse(yield_first_visit=False, yield_visited=True, raise_on_visited=True):
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
    
    def traverse(self, yield_first_visit=True, yield_visited=False, raise_on_visited=True):
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
    
    def attr_depth(self):
        # Always take length of shortest path to root
        for n in self.traverse(raise_on_visited=False):
            n.d = n.depth
    
    def attr_label(self):
        for n, label in self.nodes_df.reset_index().values():
            self.nodes[n].label = label
    
    def make_tree(self, mode='most_popular_parent'):
        if 'popular_parent' in mode:
            self.__attr_nr_leafs()
        for n in self.nodes.values():
            if mode == 'most_popular_parent':
                if len(n.parents) > 1:
                    popular_parent = n.parents[0]
                    for par in n.parents[1:]:
                        if par.w > popular_parent.w:
                            popular_parent.pop_child(n)
                            popular_parent = par
                        else:
                            par.pop_child(n)
                    n.parents = [popular_parent]
            elif mode == 'least_popular_parent':
                if len(n.parents) > 1:
                    popular_parent = n.parents[0]
                    for par in n.parents[1:]:
                        if par.w < popular_parent.w:
                            popular_parent.pop_child(n)
                            popular_parent = par
                        else:
                            par.pop_child(n)
                    n.parents = [popular_parent]
            elif mode == 'random':
                if len(n.parents) > 1:
                    keep_i = np.random.choice(len(n.parents))
                    for i, p in enumerate(n.parents):
                        if i == keep_i:
                            n.parents == [p]
                        else:
                            p.pop_child(n)
    
    def del_parent_store(self):
        for n in self.nodes.values():
            n.parents = []
        
    def add_parent_store(self):
        for n in self.nodes.values():
            for c in n.children:
                c.add_parent(n)
    
    def _del_node(self, n):
        # reconnect
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

    def filter(self, constraint, verbose=False, raise_on_visited=True, traverse=False):
        N = 0
        if traverse:
            for n in self.traverse(yield_first_visit=True, yield_visited=False, raise_on_visited=raise_on_visited):
                if constraint(n):
                    N += 1
                    self._del_node(n)
        else:
            nodes = list(self.nodes.values()).copy()
            for n in nodes:
                if constraint(n):
                    N += 1
                    self._del_node(n)
        if verbose:
            logger.log(f"Removed {N} nodes.")
    
    # def compact(self, verbose=False):
    #     self.filter(lambda n : len(n.children) == 1, verbose=verbose)

    def remove_other_roots(self, verbose=False):
        for n in self.traverse(raise_on_visited=False):
            n.important = True
        self.filter(lambda n : not n.important, verbose=verbose, traverse=False)
        for n in self.traverse(raise_on_visited=False):
            n.important = False

    # def remove_non_clinical_finding(self):
    #     for name in [272379006, 243796009, 71388002]:
    #         node = self.nodes[name]
    #         self.root.pop_child(node)
    #     self.remove_other_roots()

    def keep_names(self, names, verbose=False):
        C = 0
        for name in names:
            if name not in self.nodes:
                C += 1
            else:
                self.nodes[name].important = True
        if C > 0:
            logger.warn(f"Warning, {C} names were not found in dag.")
        self.filter(lambda n : not n.has_important, verbose=verbose, traverse=False)
        for name in names:
            if name in self.nodes:
                self.nodes[name].important = False
        self._filter_df()
    
    def _filter_df(self):
        list_nodes = list(self.nodes)
        self.nodes_df = self.nodes_df[np.isin(self.nodes_df['name'], list_nodes)]
        self.edges_df = self.edges_df[np.isin(self.edges_df['parent'], list_nodes)]
        self.edges_df = self.edges_df[np.isin(self.edges_df['child'], list_nodes)]

    def __len__(self):
        return len(self.nodes)

    def _reset_prediction(self):
        for reaction in self.prediction_df.index.values():
            self.nodes[reaction].pred = None

    def set_prediction(self, prediction_df, discard_others=False):
        assert (prediction_df.columns == ['incidence','frequency']).all()
        assert prediction_df.index.name == 'snomed_reaction'
        if self.prediction_df is not None:
            self._reset_prediction()

        self.prediction_df = prediction_df
        for reaction, incidence, frequency in prediction_df.reset_index().values():
            n = self.nodes[reaction]
            n.pred = (incidence, frequency)

        if discard_others:
            self.keep_names(prediction_df.index().values)


    def copy(self, init=True):
        # Using dataframe
        if init:
            return DAG(self.nodes_df, self.edges_df)

        # Using deepcopy
        else:
            self.del_parent_store()
            copied = copy.deepcopy(self)
            self.add_parent_store()
            copied.add_parent_store()
            return copied


def load_dag():
    clinical_finding_edges = load_data('clinical_finding_edges')
    clinical_finding_nodes = load_data('clinical_finding_nodes')
    T = DAG(clinical_finding_nodes, clinical_finding_edges)
    return T