import numpy as np
import pandas as pd
import logging
from pathlib import Path

from visualize_events.data import load_data

logger = logging.getLogger()

_PARENTS_MISSING = "Some parent nodes not found as dataframe indices"

class Node:
    def __init__(self, node, name):
        self.node = node
        self.name = name + 0
        assert str(self.name) == str(int(name))
        self.parents = []
        self.children = []
        self.important = False
        self.w = None # Can hold weight ?
        self.d = None # Can hold depth value
        self.label = None # Can hold label of concept
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
    
    def del_subtree(self):
        for par in self.parents:
            par.pop_child(self)
        for c in self.children:
            c.del_subtree()
        del c # this does not maintain tree's concept_to_node
    
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

        
class Tree:
    _WARNED = False

    def __init__(self, concepts, root):
        if isinstance(concepts, pd.DataFrame):
            root_node = Node(None, root)
            self.root = root_node
            self.concept_to_node = {root: root_node}
            self.df = concepts
            for name in self.df.index.values:
                self._expand_df(int(name))
        else:
            # Old initialization
            self.root = root
            self.concept_to_node = {self.root.name: root}
            for concept in concepts:
                if concept is not None:
                    self._expand(concept)
    
    @property
    def height(self):
        return self.root.height
    
    def _expand(self, concept):
        # start higher up if we are newly making this node
        if int(concept.name) in self.concept_to_node:
            return
        for par in concept.parents:
            if int(par.name) not in self.concept_to_node:
                self._expand(par)
        node = Node(concept, concept.name)
        self.concept_to_node[node.name] = node
        for par in concept.parents:
            par_node = self.concept_to_node[int(par.name)]
            if node not in par_node.children:
                par_node.children.append(node)
            if par_node not in node.parents:
                node.parents.append(par_node)

    def _expand_df(self, name):
        assert isinstance(name, int)
        if name in self.concept_to_node:
            return
        node = Node(None, name)
        self.concept_to_node[name] = node
        parents = []
        for par in [int(p) for p in self.df.loc[name, 'parents'].split(',')]:
            if par not in self.df.index.values:
                if not self._WARNED:
                    logger.warning(_PARENTS_MISSING)
                    self._WARNED = True
            else:
                parents.append(par)
        for par in parents:
            if par not in self.concept_to_node:
                self._expand_df(par)
        for par in parents:
            par_node = self.concept_to_node[par]
            if node not in par_node.children:
                par_node.children.append(node)
            if par_node not in node.parents:
                node.parents.append(par_node)

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
            n = to_visit.pop(0)
            if n in visited:
                if yield_visited:
                    yield n
                if raise_on_visited:
                    raise Exception("visited", n)
            else:
                if yield_first_visit:
                    yield n
                visited.add(n)
            to_visit.extend(n.children)

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
        for n in self.concept_to_node.values():
            n.label = self.df.loc[n.name, 'label']
    
    def make_tree(self, mode='most_popular_parent'):
        if 'popular_parent' in mode:
            self.__attr_nr_leafs()
        for n in self.traverse(yield_first_visit=False, yield_visited=True, raise_on_visited=False):
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
                keep_i = np.random.choice(len(n.parents))
                for i, p in enumerate(n.parents):
                    if i == keep_i:
                        n.parents == [p]
                    else:
                        p.pop_child(n)
    
    def del_parent_store(self):
        for n in self.traverse(raise_on_visited=False):
            n.parents = []
        
    def add_parent_store(self):
        for n in self.traverse(raise_on_visited=False):
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
        del self.concept_to_node[n.name]
        del n

    def filter(self, constraint, verbose=False, raise_on_visited=True, traverse=True):
        N = 0
        if traverse:
            for n in self.traverse(yield_first_visit=True, yield_visited=False, raise_on_visited=raise_on_visited):
                if constraint(n):
                    N += 1
                    self._del_node(n)
        else:
            nodes = list(self.concept_to_node.values()).copy()
            for n in nodes:
                if constraint(n):
                    N += 1
                    self._del_node(n)
        if verbose:
            print(f"Removed {N} nodes.")
    
    def compact(self, verbose=False):
        self.filter(lambda n : len(n.children) == 1, verbose=verbose)

    def remove_other_roots(self, verbose=False):
        for n in self.traverse(raise_on_visited=False):
            n.important = True
        self.filter(lambda n : not n.important, verbose=verbose, traverse=False)
        for n in self.traverse(raise_on_visited=False):
            n.important = False

    def remove_non_clinical_finding(self):
        for name in [272379006, 243796009, 71388002]:
            node = self.concept_to_node[name]
            self.root.pop_child(node)
        self.remove_other_roots()

    def keep_concepts(self, concepts, verbose=False):
        C = 0
        for concept in concepts:
            if int(concept) not in self.concept_to_node:
                C += 1
            else:
                self.concept_to_node[int(concept)].important = True
        print(f"Warning, {C} concepts were not found in tree.")
        self.filter(lambda n : not n.has_important, verbose=verbose, traverse=False)
        for concept in concepts:
            if int(concept) in self.concept_to_node:
                self.concept_to_node[int(concept)].important = False

    def __len__(self):
        return len(self.concept_to_node)

def load_tree():
    snomed_mapping = load_data('snomed_mapping')
    root_name = snomed_mapping[snomed_mapping['parents'].isna()].index.values[0]
    T = Tree(snomed_mapping, root_name)

    # path = Path(__file__)
    # data = path.parent / 'snomed_tree_pickled'
    # with open(str(data), 'rb') as f:
    #     T = pickle.load(f)
    # T.add_parent_store()

    # T.attr_depth()
    # T.attr_label()
    return T